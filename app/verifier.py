"""独立复核器：用纯 Python 重新核对求解器产出的逐段名单。

复核维度：
  1. 段数、每段恰好 5 名名单内且不重复的球员；
  2. 每段包含全部已确认球员；
  3. 逐人累计秒数与官方秒数完全一致；
  4. 相邻段实际变动人次 == 2 * 记录换下人数；
  5. 重新计算目标五人同场秒数与净胜分。
"""
from __future__ import annotations

from functools import lru_cache
from typing import Union

from .schemas import (
    LINEUP_SIZE,
    RosterCheckRequest,
    RosterReconcileRequest,
)


def _as_request(req: Union[RosterCheckRequest, dict]) -> RosterCheckRequest:
    if isinstance(req, RosterCheckRequest):
        return req
    return RosterCheckRequest.model_validate(req)


def verify_assignments(
    req: Union[RosterCheckRequest, dict], assignments: list[list[int]]
) -> list[str]:
    req = _as_request(req)
    errors: list[str] = []
    roster = set(req.players)

    if len(assignments) != len(req.segments):
        errors.append(f"段数不符：期望 {len(req.segments)}，实际 {len(assignments)}")
        return errors

    for t, lineup in enumerate(assignments):
        tag = f"第 {t + 1} 段"
        if len(lineup) != LINEUP_SIZE:
            errors.append(f"{tag}在场人数为 {len(lineup)}，应为 5")
        if len(set(lineup)) != len(lineup):
            errors.append(f"{tag}存在重复球员: {lineup}")
        unknown = set(lineup) - roster
        if unknown:
            errors.append(f"{tag}出现名单外球员: {sorted(unknown)}")
        missing = set(req.segments[t].confirmed_players) - set(lineup)
        if missing:
            errors.append(f"{tag}缺少已确认球员: {sorted(missing)}")

    # 换人记录
    for t in range(1, len(assignments)):
        prev, cur = set(assignments[t - 1]), set(assignments[t])
        moved = len(prev.symmetric_difference(cur))
        expected = 2 * req.segments[t].players_substituted
        if moved != expected:
            errors.append(
                f"第 {t + 1} 段实际变动 {moved} 人次，"
                f"换人记录要求 {expected} 人次"
            )

    # 官方累计秒数
    totals = compute_player_totals(req, assignments)
    for p in req.players:
        if totals.get(p, 0) != req.official_seconds[p]:
            errors.append(
                f"球员 {p} 累计秒数 {totals.get(p, 0)} "
                f"与官方秒数 {req.official_seconds[p]} 不一致"
            )
    return errors


def compute_player_totals(
    req: Union[RosterCheckRequest, dict], assignments: list[list[int]]
) -> dict[int, int]:
    req = _as_request(req)
    totals = {p: 0 for p in req.players}
    for t, lineup in enumerate(assignments):
        for p in lineup:
            totals[p] += req.segments[t].duration_seconds
    return totals


def compute_shared_seconds(
    req: Union[RosterCheckRequest, dict], assignments: list[list[int]]
) -> int:
    req = _as_request(req)
    target = set(req.target_lineup)
    return sum(
        req.segments[t].duration_seconds
        for t, lineup in enumerate(assignments)
        if target <= set(lineup)
    )


def compute_net_points(
    req: Union[RosterCheckRequest, dict], assignments: list[list[int]]
) -> int:
    req = _as_request(req)
    target = set(req.target_lineup)
    return sum(
        req.segments[t].team_net_points
        for t, lineup in enumerate(assignments)
        if target <= set(lineup)
    )


def actual_substitution_counts(assignments: list[list[int]]) -> list[int]:
    """逐段实际换下人数（首段为 0）。"""
    counts = [0]
    for t in range(1, len(assignments)):
        prev, cur = set(assignments[t - 1]), set(assignments[t])
        counts.append(len(prev - cur))
    return counts


# --------------------------------------------------------------------------- #
# 记分归属的独立复核（不复用 attribution.py 的 DP，改用集合记忆化搜索）
# --------------------------------------------------------------------------- #
def _as_reconcile(
    req: Union[RosterReconcileRequest, dict],
) -> RosterReconcileRequest:
    if isinstance(req, RosterReconcileRequest):
        return req
    return RosterReconcileRequest.model_validate(req)


def verify_attribution(
    req: Union[RosterReconcileRequest, dict], entry_segments: list[int]
) -> list[str]:
    """校验一套「每笔流水 → 片段号」的完整归属。"""
    req = _as_reconcile(req)
    errors: list[str] = []
    T = len(req.segments)

    if len(entry_segments) != len(req.scoring_entries):
        return [
            f"归属笔数 {len(entry_segments)} 与流水笔数 "
            f"{len(req.scoring_entries)} 不一致"
        ]

    prev = 1
    for idx, (seg, entry) in enumerate(zip(entry_segments, req.scoring_entries)):
        tag = f"第 {idx + 1} 笔记分"
        if not (1 <= seg <= T):
            errors.append(f"{tag}归属片段号 {seg} 超出 1~{T}")
            continue
        if seg not in entry.candidate_segments:
            errors.append(
                f"{tag}归入片段 {seg}，但候选片段为 {entry.candidate_segments}"
            )
        if idx > 0 and seg < prev:
            errors.append(f"{tag}归入片段 {seg}，片段号较前一笔 {prev} 倒退")
        prev = seg

    for cp in req.score_checkpoints:
        limit = cp.segment_number
        team = sum(
            e.team_points
            for e, seg in zip(req.scoring_entries, entry_segments)
            if seg <= limit
        )
        opp = sum(
            e.opponent_points
            for e, seg in zip(req.scoring_entries, entry_segments)
            if seg <= limit
        )
        if (team, opp) != (
            cp.team_cumulative_points,
            cp.opponent_cumulative_points,
        ):
            errors.append(
                f"核对点 {limit} 实际累计 ({team},{opp})，"
                f"记录要求 ({cp.team_cumulative_points},"
                f"{cp.opponent_cumulative_points})"
            )
    return errors


def attribution_options(
    req: Union[RosterReconcileRequest, dict],
) -> dict:
    """独立枚举归属（记忆化搜索），返回可行性与逐段可达净分集合。"""
    req = _as_reconcile(req)
    T = len(req.segments)
    entries = req.scoring_entries
    N = len(entries)
    cps = {cp.segment_number: cp for cp in req.score_checkpoints}

    # 用 (片段, 起笔) 到 (片段, 止笔) 的 DP 求逐段净分集合
    prefix_team = [0] * (N + 1)
    prefix_opp = [0] * (N + 1)
    for i, e in enumerate(entries):
        prefix_team[i + 1] = prefix_team[i] + e.team_points
        prefix_opp[i + 1] = prefix_opp[i] + e.opponent_points

    # run_ok[t][i]：从第 i 笔（0 起）起连续允许归入片段 t 的最长笔数
    run_ok: list[list[int]] = [[0] * (N + 1) for _ in range(T + 1)]
    for t in range(1, T + 1):
        run_ok[t][N] = N
        for i in range(N - 1, -1, -1):
            run_ok[t][i] = run_ok[t][i + 1] if t in entries[i].candidate_segments else i

    @lru_cache(maxsize=None)
    def fwd(t: int, i: int) -> bool:
        """片段 1..t 恰好消费前 i 笔且满足其间核对点。"""
        if t == 0:
            return i == 0
        cp = cps.get(t)
        if cp is not None and (
            prefix_team[i] != cp.team_cumulative_points
            or prefix_opp[i] != cp.opponent_cumulative_points
        ):
            return False
        for a in range(i, -1, -1):
            if not fwd(t - 1, a):
                continue
            if i <= run_ok[t][a]:
                return True
        return False

    @lru_cache(maxsize=None)
    def bwd(t: int, i: int) -> bool:
        """片段 t+1..T 恰好消费第 i..N 笔且满足其间核对点。"""
        if t == T:
            return i == N
        cp = cps.get(t + 1)
        for b in range(i, N + 1):
            if b > run_ok[t + 1][i]:
                break
            if cp is not None and (
                prefix_team[b] != cp.team_cumulative_points
                or prefix_opp[b] != cp.opponent_cumulative_points
            ):
                continue
            if bwd(t + 1, b):
                return True
        return False

    possible: list[set[int]] = [set() for _ in range(T)]
    for t in range(1, T + 1):
        cp = cps.get(t)
        for i in range(N + 1):
            if not fwd(t - 1, i):
                continue
            for j in range(i, run_ok[t][i] + 1):
                if cp is not None and (
                    prefix_team[j] != cp.team_cumulative_points
                    or prefix_opp[j] != cp.opponent_cumulative_points
                ):
                    continue
                if not bwd(t, j):
                    continue
                possible[t - 1].add(
                    (prefix_team[j] - prefix_team[i])
                    - (prefix_opp[j] - prefix_opp[i])
                )
    feasible = bwd(0, 0) and all(possible[t] for t in range(T))
    return {
        "feasible": feasible,
        "possible_nets": [set(s) for s in possible],
    }


def verify_reconciled_request_body(body: dict, nets: list[int]) -> list[str]:
    """校验回填后的请求体：team_net_points 与给定净分一致，且结构可被原接口接受。"""
    errors: list[str] = []
    segments = body.get("segments", [])
    if len(segments) != len(nets):
        errors.append("回填请求体的片段数与净分数量不一致")
    for idx, (seg, net) in enumerate(zip(segments, nets)):
        if seg.get("team_net_points") != net:
            errors.append(
                f"回填请求体第 {idx + 1} 段 team_net_points "
                f"{seg.get('team_net_points')} 与净分 {net} 不一致"
            )
    try:
        RosterCheckRequest.model_validate(body)
    except Exception as exc:  # noqa: BLE001 - 复核器只汇报，不抛出
        errors.append(f"回填请求体无法通过 /roster/check 的结构校验: {exc}")
    return errors
