"""记分流水 → 换人片段的归属对账。

核心模型
--------
记分流水按时间有序（`scoring_entries[0..N-1]`），每一笔只能整体归入其
`candidate_segments` 中的一个片段，且片段号沿流水不得倒退。由此可推出：

    归入同一片段的流水笔必然在流水中连续，
    一套完整归属等价于把流水切成 T 个连续块（块可以为空），
    第 t 块整体归入片段 t（块内每一笔都允许片段 t）。

因此状态只需 `(已处理片段数 t, 已消费流水笔数 i)`：

    prefix_team[i] / prefix_opponent[i]  前 i 笔的累计得分；
    核对点（片段 t 末）要求 prefix_*[i] 同时等于记录的双方累计。

预处理 `run_end[t][i]`（从第 i 笔起，连续多少笔允许归入片段 t）后，
前向 / 后向可达性都是 O(T·N²) 的网格 DP，足以覆盖 T≤16、N≤120。

对每个片段 s，枚举所有同时满足前向、后向可达的块 (i, j)，
即可得到该片段在所有合规归属下可能取得的净分集合。

- 每个片段净分集合都是单点 → `resolved`，回填 team_net_points；
- 存在片段净分集合不止一个 → `ambiguous`，列出全部受影响片段，
  并重建两套使某一受影响片段净分不同的完整归属；
- 不存在合规归属 → `score_conflict`，给出精确冲突定位。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import RosterReconcileRequest
from .verifier import (
    attribution_options,
    verify_attribution,
    verify_reconciled_request_body,
)

STATUS_RESOLVED = "resolved"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_CONFLICT = "score_conflict"

STATUS_ZH = {
    STATUS_RESOLVED: "逐段净分唯一确定",
    STATUS_AMBIGUOUS: "逐段净分不唯一",
    STATUS_CONFLICT: "比分记录冲突",
}


@dataclass
class AttributionPlan:
    """一套完整归属（每笔流水 → 片段号）及其逐段汇总。"""

    label: str
    entry_segments: list[int]
    segment_team_points: list[int]
    segment_opponent_points: list[int]
    segment_net_points: list[int]
    checkpoint_results: list[dict] = field(default_factory=list)


@dataclass
class ReconcileOutcome:
    status: str
    conflicts: list[dict] = field(default_factory=list)
    resolved_team_net_points: list[int] | None = None
    affected_segments: list[int] = field(default_factory=list)
    witness_segment: int | None = None
    plans: list[AttributionPlan] = field(default_factory=list)
    request_body: dict | None = None
    verification: str | None = None
    verification_errors: list[str] = field(default_factory=list)

    @property
    def status_label_zh(self) -> str:
        return STATUS_ZH[self.status]


class ScoreReconciler:
    def __init__(self, req: RosterReconcileRequest) -> None:
        self.req = req
        self.T = len(req.segments)
        self.candidates: list[set[int]] = [
            set(e.candidate_segments) for e in req.scoring_entries
        ]
        self.N = len(self.candidates)
        self.checkpoints = {
            cp.segment_number: (
                cp.team_cumulative_points,
                cp.opponent_cumulative_points,
            )
            for cp in req.score_checkpoints
        }
        self.prefix_team = [0] * (self.N + 1)
        self.prefix_opp = [0] * (self.N + 1)
        for i, e in enumerate(req.scoring_entries):
            self.prefix_team[i + 1] = self.prefix_team[i] + e.team_points
            self.prefix_opp[i + 1] = self.prefix_opp[i] + e.opponent_points

        # run_end[t][i]：从第 i 笔（0 起）开始，连续多少笔都允许归入片段 t。
        self.run_end: list[list[int]] = [[0] * (self.N + 1) for _ in range(self.T + 1)]
        for t in range(1, self.T + 1):
            self.run_end[t][self.N] = self.N
            for i in range(self.N - 1, -1, -1):
                if t in self.candidates[i]:
                    self.run_end[t][i] = self.run_end[t][i + 1]
                else:
                    self.run_end[t][i] = i

        self.fwd: list[list[bool]] = [
            [False] * (self.N + 1) for _ in range(self.T + 1)
        ]
        self.pred: list[list[int | None]] = [
            [None] * (self.N + 1) for _ in range(self.T + 1)
        ]
        self.bwd: list[list[bool]] = [
            [False] * (self.N + 1) for _ in range(self.T + 1)
        ]
        self.next_state: list[list[int | None]] = [
            [None] * (self.N + 1) for _ in range(self.T)
        ]

    # ------------------------------------------------------------------ #
    # 基础判定
    # ------------------------------------------------------------------ #
    def _checkpoint_allows(self, segment: int, consumed: int) -> bool:
        cp = self.checkpoints.get(segment)
        if cp is None:
            return True
        return (
            self.prefix_team[consumed] == cp[0]
            and self.prefix_opp[consumed] == cp[1]
        )

    def _forward(self) -> None:
        self.fwd[0][0] = True
        for t in range(1, self.T + 1):
            for j in range(self.N + 1):
                if not self._checkpoint_allows(t, j):
                    continue
                for i in range(j + 1):
                    if not self.fwd[t - 1][i]:
                        continue
                    if j <= self.run_end[t][i]:
                        self.fwd[t][j] = True
                        self.pred[t][j] = i
                        break

    def _backward(self) -> None:
        # 处理完 T 个片段后必须恰好消费全部流水，且 T 段末核对点（若有）成立。
        self.bwd[self.T][self.N] = self._checkpoint_allows(self.T, self.N)
        for t in range(self.T - 1, -1, -1):
            for i in range(self.N + 1):
                if t > 0 and not self._checkpoint_allows(t, i):
                    continue
                for j in range(i, self.run_end[t + 1][i] + 1):
                    if self.bwd[t + 1][j]:
                        self.bwd[t][i] = True
                        self.next_state[t][i] = j
                        break

    def _segment_blocks(self) -> list[list[tuple[int, int, int]]]:
        """逐片段列出所有合规块 (i, j, net)（含重复 net 的代表边）。"""
        blocks: list[list[tuple[int, int, int]]] = [[] for _ in range(self.T + 1)]
        for t in range(1, self.T + 1):
            for i in range(self.N + 1):
                if not self.fwd[t - 1][i]:
                    continue
                for j in range(i, self.run_end[t][i] + 1):
                    if not self.bwd[t][j]:
                        continue
                    net = (self.prefix_team[j] - self.prefix_team[i]) - (
                        self.prefix_opp[j] - self.prefix_opp[i]
                    )
                    blocks[t].append((i, j, net))
        return blocks

    # ------------------------------------------------------------------ #
    # 冲突诊断
    # ------------------------------------------------------------------ #
    def _diagnose_precheck(self, diag: list[dict]) -> None:
        for idx, cand in enumerate(self.candidates):
            if not cand:
                diag.append(
                    {
                        "code": "ENTRY_HAS_NO_CANDIDATE_SEGMENT",
                        "message": f"第 {idx + 1} 笔记分没有任何候选片段，无法归属",
                        "entry_index": idx + 1,
                        "team_points": self.req.scoring_entries[idx].team_points,
                        "opponent_points": self.req.scoring_entries[idx].opponent_points,
                    }
                )
        if diag:
            return

        # 忽略核对点时，片段号不倒退的归属一定存在当且仅当贪心（每笔取可选最小片段）成功。
        cur = 1
        for idx, cand in enumerate(self.candidates):
            feasible = sorted(s for s in cand if s >= cur)
            if not feasible:
                diag.append(
                    {
                        "code": "ENTRY_SEGMENT_ORDER_INFEASIBLE",
                        "message": (
                            f"第 {idx + 1} 笔记分的候选片段 {sorted(cand)} "
                            f"都小于已到达的片段号 {cur}，片段号无法沿流水不倒退"
                        ),
                        "entry_index": idx + 1,
                        "candidate_segments": sorted(cand),
                        "minimum_segment_reached_by_previous_entries": cur,
                    }
                )
                return
            cur = feasible[0]

        ordered = sorted(self.checkpoints.items())
        for (s1, (team1, opp1)), (s2, (team2, opp2)) in zip(ordered, ordered[1:]):
            if team2 < team1 or opp2 < opp1:
                diag.append(
                    {
                        "code": "CHECKPOINT_CUMULATIVE_DECREASE",
                        "message": (
                            f"核对点 {s2} 的累计得分 ({team2},{opp2}) 小于"
                            f"核对点 {s1} 的累计 ({team1},{opp1})；得分只增不减"
                        ),
                        "earlier_segment": s1,
                        "later_segment": s2,
                        "earlier_cumulative": {
                            "team": team1,
                            "opponent": opp1,
                        },
                        "later_cumulative": {"team": team2, "opponent": opp2},
                    }
                )
                return

        if self.T in self.checkpoints:
            team, opp = self.checkpoints[self.T]
            total = (self.prefix_team[self.N], self.prefix_opp[self.N])
            if (team, opp) != total:
                diag.append(
                    {
                        "code": "CHECKPOINT_FINAL_TOTAL_MISMATCH",
                        "message": (
                            f"核对点 {self.T} 位于最后一段，其累计 ({team},{opp}) "
                            f"必须等于全部记分合计 ({total[0]},{total[1]})"
                        ),
                        "segment_number": self.T,
                        "expected_cumulative": {"team": team, "opponent": opp},
                        "stream_total": {"team": total[0], "opponent": total[1]},
                    }
                )
                return

        for s in sorted(self.checkpoints):
            team, opp = self.checkpoints[s]
            attainable = [
                i
                for i in range(self.N + 1)
                if self.prefix_team[i] == team and self.prefix_opp[i] == opp
            ]
            if not attainable:
                nearest = min(
                    range(self.N + 1),
                    key=lambda i: abs(self.prefix_team[i] - team)
                    + abs(self.prefix_opp[i] - opp),
                )
                diag.append(
                    {
                        "code": "CHECKPOINT_NOT_ATTAINABLE",
                        "message": (
                            f"核对点 {s} 要求片段末累计 ({team},{opp})，"
                            "但流水没有哪个前缀的累计得分同时等于该值"
                        ),
                        "segment_number": s,
                        "expected_cumulative": {"team": team, "opponent": opp},
                        "nearest_prefix": {
                            "entries_consumed": nearest,
                            "team": self.prefix_team[nearest],
                            "opponent": self.prefix_opp[nearest],
                        },
                        "stream_total": {
                            "team": self.prefix_team[self.N],
                            "opponent": self.prefix_opp[self.N],
                        },
                    }
                )
                return

    def _diagnose_dp_failure(self) -> list[dict]:
        """DP 不可行且预检未定位时的补充诊断。"""
        # 找到带核对点的、首个前向可达状态为空的片段
        for t in range(1, self.T + 1):
            if any(self.fwd[t]):
                continue
            if t in self.checkpoints:
                team, opp = self.checkpoints[t]
                return [
                    {
                        "code": "CHECKPOINT_UNREACHABLE_AFTER_SEGMENT",
                        "message": (
                            f"核对点 {t} 之前的片段无法同时容纳流水前缀并满足"
                            "片段号不倒退，或该核对点累计无匹配前缀"
                        ),
                        "segment_number": t,
                        "expected_cumulative": {"team": team, "opponent": opp},
                    }
                ]
            return [
                {
                    "code": "SCORE_ATTRIBUTION_INFEASIBLE",
                    "message": (
                        f"片段 1~{t} 的候选片段约束与片段号不倒退要求"
                        "无法同时满足"
                    ),
                    "first_infeasible_segment": t,
                }
            ]
        if not self.fwd[self.T][self.N]:
            return [
                {
                    "code": "SCORE_ATTRIBUTION_INFEASIBLE",
                    "message": "全部流水无法在片段 1~T 内按片段号不倒退地完整归属",
                    "stream_entries": self.N,
                    "segments": self.T,
                }
            ]
        return [
            {
                "code": "SCORE_ATTRIBUTION_INFEASIBLE",
                "message": "记分流水与候选片段、核对点无法同时成立",
            }
        ]

    # ------------------------------------------------------------------ #
    # 归属重建
    # ------------------------------------------------------------------ #
    def _build_plan(
        self, label: str, witness_segment: int, i: int, j: int
    ) -> AttributionPlan:
        assert self.fwd[witness_segment - 1][i]
        assert self.bwd[witness_segment][j]

        # 前向：状态 (0,0) → (witness_segment-1, i)
        forward_states: list[tuple[int, int]] = [(witness_segment - 1, i)]
        t, cur = witness_segment - 1, i
        while t > 0:
            p = self.pred[t][cur]
            assert p is not None
            forward_states.append((t - 1, p))
            t, cur = t - 1, p
        forward_states.reverse()

        entry_segments = [0] * self.N
        for seg in range(1, witness_segment):
            a = forward_states[seg - 1][1]
            b = forward_states[seg][1]
            for k in range(a, b):
                entry_segments[k] = seg
        for k in range(i, j):
            entry_segments[k] = witness_segment

        # 后向：(witness_segment, j) → (T, N)
        t, cur = witness_segment, j
        while t < self.T:
            nxt = self.next_state[t][cur]
            assert nxt is not None
            for k in range(cur, nxt):
                entry_segments[k] = t + 1
            t, cur = t + 1, nxt
        assert cur == self.N

        assert all(s > 0 for s in entry_segments), "重建归属未覆盖全部流水"
        return self._summarize(label, entry_segments)

    def _summarize(self, label: str, entry_segments: list[int]) -> AttributionPlan:
        seg_team = [0] * self.T
        seg_opp = [0] * self.T
        for k, seg in enumerate(entry_segments):
            e = self.req.scoring_entries[k]
            seg_team[seg - 1] += e.team_points
            seg_opp[seg - 1] += e.opponent_points
        seg_net = [seg_team[t] - seg_opp[t] for t in range(self.T)]

        checkpoint_results = []
        for cp in sorted(self.req.score_checkpoints, key=lambda c: c.segment_number):
            upto = cp.segment_number
            actual_team = sum(seg_team[:upto])
            actual_opp = sum(seg_opp[:upto])
            checkpoint_results.append(
                {
                    "segment_number": upto,
                    "expected_team_cumulative_points": cp.team_cumulative_points,
                    "actual_team_cumulative_points": actual_team,
                    "expected_opponent_cumulative_points": cp.opponent_cumulative_points,
                    "actual_opponent_cumulative_points": actual_opp,
                    "matches": (
                        actual_team == cp.team_cumulative_points
                        and actual_opp == cp.opponent_cumulative_points
                    ),
                }
            )
        return AttributionPlan(
            label=label,
            entry_segments=entry_segments,
            segment_team_points=seg_team,
            segment_opponent_points=seg_opp,
            segment_net_points=seg_net,
            checkpoint_results=checkpoint_results,
        )

    def _resolved_body(self, nets: list[int]) -> dict:
        return {
            "players": list(self.req.players),
            "target_lineup": list(self.req.target_lineup),
            "minimum_seconds": self.req.minimum_seconds,
            "official_seconds": {
                str(p): self.req.official_seconds[p] for p in self.req.players
            },
            "segments": [
                {
                    "duration_seconds": seg.duration_seconds,
                    "confirmed_players": list(seg.confirmed_players),
                    "players_substituted": seg.players_substituted,
                    "team_net_points": nets[t],
                }
                for t, seg in enumerate(self.req.segments)
            ],
        }

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #
    def run(self) -> ReconcileOutcome:
        diag: list[dict] = []
        self._diagnose_precheck(diag)
        if diag:
            return self._finish_conflict(diag)

        self._forward()
        self._backward()
        if not self.bwd[0][0] or not self.fwd[self.T][self.N]:
            return self._finish_conflict(self._diagnose_dp_failure())

        blocks = self._segment_blocks()
        possible_nets: list[set[int]] = [set() for _ in range(self.T)]
        representative: list[dict[int, tuple[int, int]]] = [
            {} for _ in range(self.T)
        ]
        for t in range(1, self.T + 1):
            for i, j, net in blocks[t]:
                possible_nets[t - 1].add(net)
                representative[t - 1].setdefault(net, (i, j))

        # 防御：每个片段至少应有一个可达块
        if any(not s for s in possible_nets):
            return self._finish_conflict(
                [
                    {
                        "code": "SCORE_ATTRIBUTION_INFEASIBLE",
                        "message": "存在片段在所有合规归属下都无法对应任何连续流水块",
                    }
                ]
            )

        affected = [t + 1 for t in range(self.T) if len(possible_nets[t]) > 1]
        if not affected:
            nets = [next(iter(s)) for s in possible_nets]
            # 任取一条满足 (片段 T, 消费 N) 的路径作为见证归属
            plan = self._build_plan(
                "唯一净分见证归属",
                self.T,
                self.pred[self.T][self.N],
                self.N,
            )
            return self._finish_resolved(nets, plan)

        witness_segment = affected[0]
        values = sorted(possible_nets[witness_segment - 1])
        plan_a = self._build_plan(
            "方案A",
            witness_segment,
            *representative[witness_segment - 1][values[0]],
        )
        plan_b = self._build_plan(
            "方案B",
            witness_segment,
            *representative[witness_segment - 1][values[-1]],
        )
        return self._finish_ambiguous(affected, witness_segment, plan_a, plan_b)

    # ------------------------------------------------------------------ #
    # 结果收尾与独立复核
    # ------------------------------------------------------------------ #
    def _finish_conflict(self, conflicts: list[dict]) -> ReconcileOutcome:
        options = attribution_options(self.req)
        errs = []
        if options["feasible"]:
            errs.append("独立复核认为存在合规归属，但主流程判定冲突")
        return ReconcileOutcome(
            status=STATUS_CONFLICT,
            conflicts=conflicts,
            verification=(
                "独立复核确认：不存在同时满足候选片段、片段号不倒退与全部核对点的归属"
                if not errs
                else "独立复核发现冲突判定不一致"
            ),
            verification_errors=errs,
        )

    def _finish_resolved(
        self, nets: list[int], plan: AttributionPlan
    ) -> ReconcileOutcome:
        body = self._resolved_body(nets)
        errors = verify_attribution(self.req, plan.entry_segments)
        options = attribution_options(self.req)
        if not options["feasible"]:
            errors.append("独立复核认为该记分流水不存在合规归属")
        for t, values in enumerate(options["possible_nets"], start=1):
            if values != {nets[t - 1]}:
                errors.append(
                    f"第 {t} 段净分独立复核为 {sorted(values)}，"
                    f"与唯一值 {nets[t - 1]} 不一致"
                )
        errors.extend(verify_reconciled_request_body(body, nets))
        if errors:
            return ReconcileOutcome(
                status=STATUS_RESOLVED,
                resolved_team_net_points=nets,
                plans=[plan],
                request_body=body,
                verification="独立复核发现不一致（详见 verification_errors）",
                verification_errors=errors,
            )
        return ReconcileOutcome(
            status=STATUS_RESOLVED,
            resolved_team_net_points=nets,
            plans=[plan],
            request_body=body,
            verification=(
                f"已通过独立复核：{self.N} 笔记分存在合规归属，"
                f"{self.T} 个片段的净分在所有归属下均唯一；"
                "回填后的请求体可提交 /api/v1/roster/check"
            ),
        )

    def _finish_ambiguous(
        self,
        affected: list[int],
        witness_segment: int,
        plan_a: AttributionPlan,
        plan_b: AttributionPlan,
    ) -> ReconcileOutcome:
        errors: list[str] = []
        options = attribution_options(self.req)
        expected_affected = [
            t + 1 for t, values in enumerate(options["possible_nets"]) if len(values) > 1
        ]
        if expected_affected != affected:
            errors.append(
                f"受影响片段独立复核为 {expected_affected}，与主流程 {affected} 不一致"
            )
        for plan in (plan_a, plan_b):
            errors.extend(
                f"[{plan.label}] {e}"
                for e in verify_attribution(self.req, plan.entry_segments)
            )
        if (
            plan_a.segment_net_points[witness_segment - 1]
            == plan_b.segment_net_points[witness_segment - 1]
        ):
            errors.append("两套见证归属在见证片段上的净分相同，无法区分歧义")
        if errors:
            return ReconcileOutcome(
                status=STATUS_AMBIGUOUS,
                affected_segments=affected,
                witness_segment=witness_segment,
                plans=[plan_a, plan_b],
                verification="独立复核发现不一致（详见 verification_errors）",
                verification_errors=errors,
            )
        return ReconcileOutcome(
            status=STATUS_AMBIGUOUS,
            affected_segments=affected,
            witness_segment=witness_segment,
            plans=[plan_a, plan_b],
            verification=(
                f"已通过独立复核：两套完整归属均满足候选片段、片段号不倒退与全部核对点，"
                f"并在片段 {witness_segment} 得到不同净分"
                f"（{plan_a.segment_net_points[witness_segment - 1]} vs "
                f"{plan_b.segment_net_points[witness_segment - 1]}）；"
                f"净分可能变化的片段共 {len(affected)} 个"
            ),
        )
