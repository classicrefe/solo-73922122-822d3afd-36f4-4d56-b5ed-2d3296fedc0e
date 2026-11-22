"""请求 / 响应 Pydantic 模型。

结构层面的非法输入（数量越界、ID 不存在、重复 ID 等）在本层校验，FastAPI 返回 422；
记录之间的语义冲突（秒数对不上、换人记录矛盾等）在求解阶段诊断，返回 200 + conflicts。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_PLAYERS = 12
MIN_PLAYERS = 5
LINEUP_SIZE = 5
MAX_SEGMENTS = 16
MAX_SCORING_ENTRIES = 120

VerdictStatus = Literal["guaranteed", "possible", "impossible", "infeasible_record"]
ReconcileStatus = Literal["resolved", "ambiguous", "score_conflict"]


class SegmentInput(BaseModel):
    """按换人切分的一个有序片段。"""

    duration_seconds: int = Field(..., ge=1, description="片段时长（整秒），>=1")
    confirmed_players: list[int] = Field(
        default_factory=list, description="该片段已确认在场的球员 ID"
    )
    players_substituted: int = Field(
        ..., ge=0, le=5, description="相对前段的换下人数（首段必须为 0）"
    )
    team_net_points: int = Field(
        ..., description="本队该片段得失分（净胜分，可为负）"
    )

    @field_validator("confirmed_players")
    @classmethod
    def _confirmed_unique(cls, v: list[int]) -> list[int]:
        if len(v) != len(set(v)):
            raise ValueError("confirmed_players 不能包含重复球员")
        if len(v) > LINEUP_SIZE:
            raise ValueError("单片段已确认球员不能超过 5 人")
        return v


class RosterCheckRequest(BaseModel):
    """阵容达标核对请求。"""

    players: list[int] = Field(..., description="本节进入过名单的全部球员 ID（5~12 人）")
    target_lineup: list[int] = Field(..., description="目标五人阵容 ID（恰好 5 人）")
    minimum_seconds: int = Field(..., ge=0, description="最低同场上场时长（整秒）")
    official_seconds: dict[int, int] = Field(
        ..., description="官方逐人本节累计在场秒数，键为球员 ID"
    )
    segments: list[SegmentInput] = Field(..., description="按换人切分的有序片段（1~16 段）")

    @field_validator("players")
    @classmethod
    def _players_unique(cls, v: list[int]) -> list[int]:
        if not (MIN_PLAYERS <= len(v) <= MAX_PLAYERS):
            raise ValueError(f"players 人数必须在 {MIN_PLAYERS}~{MAX_PLAYERS} 之间")
        if len(v) != len(set(v)):
            raise ValueError("players 不能包含重复球员")
        if any(p < 1 for p in v):
            raise ValueError("球员 ID 必须为正整数")
        return v

    @field_validator("target_lineup")
    @classmethod
    def _target_unique(cls, v: list[int]) -> list[int]:
        if len(v) != LINEUP_SIZE:
            raise ValueError("target_lineup 必须恰好为 5 人")
        if len(v) != len(set(v)):
            raise ValueError("target_lineup 不能包含重复球员")
        return v

    @model_validator(mode="after")
    def _cross_check(self) -> "RosterCheckRequest":
        roster = set(self.players)

        if not set(self.target_lineup) <= roster:
            raise ValueError("target_lineup 必须全部来自 players")

        if set(self.official_seconds.keys()) != roster:
            raise ValueError("official_seconds 的键必须与 players 完全一致")
        if any(v < 0 for v in self.official_seconds.values()):
            raise ValueError("official_seconds 不能为负")

        if not (1 <= len(self.segments) <= MAX_SEGMENTS):
            raise ValueError(f"segments 数量必须在 1~{MAX_SEGMENTS} 之间")

        for idx, seg in enumerate(self.segments):
            unknown = set(seg.confirmed_players) - roster
            if unknown:
                raise ValueError(f"第 {idx + 1} 段出现名单外球员: {sorted(unknown)}")
            if idx == 0 and seg.players_substituted != 0:
                raise ValueError("首段 players_substituted 必须为 0")
        return self


# --------------------------------------------------------------------------- #
# 记分流水对账
# --------------------------------------------------------------------------- #
class ReconcileSegmentInput(BaseModel):
    """未填 `team_net_points` 的片段输入（其余字段与 /roster/check 完全一致）。"""

    model_config = ConfigDict(extra="forbid")

    duration_seconds: int = Field(..., ge=1, description="片段时长（整秒），>=1")
    confirmed_players: list[int] = Field(
        default_factory=list, description="该片段已确认在场的球员 ID"
    )
    players_substituted: int = Field(
        ..., ge=0, le=5, description="相对前段的换下人数（首段必须为 0）"
    )
    team_net_points: None = Field(
        default=None,
        description="必须留空；对账接口负责回填该字段",
    )

    @field_validator("confirmed_players")
    @classmethod
    def _confirmed_unique(cls, v: list[int]) -> list[int]:
        if len(v) != len(set(v)):
            raise ValueError("confirmed_players 不能包含重复球员")
        if len(v) > LINEUP_SIZE:
            raise ValueError("单片段已确认球员不能超过 5 人")
        return v


class ScoringEntryInput(BaseModel):
    """一笔有序记分：双方各得多少分，以及这笔分可能落在哪些片段。"""

    model_config = ConfigDict(extra="forbid")

    team_points: int = Field(..., ge=0, description="本队该笔增加的分数（非负整数）")
    opponent_points: int = Field(..., ge=0, description="对方该笔增加的分数（非负整数）")
    candidate_segments: list[int] = Field(
        ...,
        description=(
            "该笔记分可能归属的片段号（1 起）。允许跳号；整笔只能归入其中一个片段。"
            "空列表表示该笔无法归属任何片段（会导致记分冲突）。"
        ),
    )

    @field_validator("candidate_segments")
    @classmethod
    def _candidates_unique(cls, v: list[int]) -> list[int]:
        if len(v) != len(set(v)):
            raise ValueError("candidate_segments 不能包含重复片段号")
        return v


class ScoreCheckpointInput(BaseModel):
    """片段末比分：该片段结束时双方本节累计得分。"""

    model_config = ConfigDict(extra="forbid")

    segment_number: int = Field(..., ge=1, description="核对点所在片段号（1 起）")
    team_cumulative_points: int = Field(
        ..., ge=0, description="该片段结束时本队本节累计得分"
    )
    opponent_cumulative_points: int = Field(
        ..., ge=0, description="该片段结束时对方本节累计得分"
    )


class RosterReconcileRequest(RosterCheckRequest):
    """记分流水对账请求：未填 team_net_points 的 roster/check 请求 + 记分流水。"""

    segments: list[ReconcileSegmentInput] = Field(
        ..., description="按换人切分的有序片段（1~16 段，暂不填 team_net_points）"
    )
    scoring_entries: list[ScoringEntryInput] = Field(
        ...,
        max_length=MAX_SCORING_ENTRIES,
        description=f"按时间顺序排列的记分流水（0~{MAX_SCORING_ENTRIES} 笔）",
    )
    score_checkpoints: list[ScoreCheckpointInput] = Field(
        default_factory=list,
        description="可选的片段末比分核对点（片段的双方本节累计得分）",
    )

    @model_validator(mode="after")
    def _reconcile_cross_check(self) -> "RosterReconcileRequest":
        T = len(self.segments)

        for idx, entry in enumerate(self.scoring_entries):
            for c in entry.candidate_segments:
                if not (1 <= c <= T):
                    raise ValueError(
                        f"第 {idx + 1} 笔记分的候选片段号 {c} 超出 1~{T} 范围"
                    )

        seen: set[int] = set()
        for cp in self.score_checkpoints:
            if cp.segment_number > T:
                raise ValueError(
                    f"核对点片段号 {cp.segment_number} 超出 1~{T} 范围"
                )
            if cp.segment_number in seen:
                raise ValueError(
                    f"核对点片段号 {cp.segment_number} 重复，每个片段末最多一个核对点"
                )
            seen.add(cp.segment_number)
        return self


class EntryAssignment(BaseModel):
    """一套完整归属中的单笔记分去向。"""

    entry_index: int = Field(..., description="记分流水序号（1 起）")
    segment_number: int
    team_points: int
    opponent_points: int
    net_points: int


class CheckpointResult(BaseModel):
    """一个核对点在给定归属下的实际累计与比对结果。"""

    segment_number: int
    expected_team_cumulative_points: int
    actual_team_cumulative_points: int
    expected_opponent_cumulative_points: int
    actual_opponent_cumulative_points: int
    matches: bool


class ScoreAttribution(BaseModel):
    """一套完整记分归属及逐段汇总。"""

    label: str
    entry_assignments: list[EntryAssignment]
    segment_team_points: list[int]
    segment_opponent_points: list[int]
    segment_net_points: list[int]
    checkpoints: list[CheckpointResult] = Field(default_factory=list)


class ReconcileResponse(BaseModel):
    status: ReconcileStatus
    status_label_zh: str
    affected_segments: list[int] = Field(
        default_factory=list,
        description="净分可能随合规归属变化的片段号（仅 ambiguous 时非空）",
    )
    witness_segment: int | None = Field(
        default=None,
        description="两套见证归属所区分的片段号（仅 ambiguous 时给出）",
    )
    resolved_team_net_points: list[int] | None = Field(
        default=None,
        description="逐段净分（本队增分 - 对方增分），仅 resolved 时给出",
    )
    request_body: dict | None = Field(
        default=None,
        description="已回填 team_net_points、可直接提交给 /api/v1/roster/check 的请求体",
    )
    witnesses: list[ScoreAttribution] = Field(
        default_factory=list,
        description="resolved 给 1 套；ambiguous 给 2 套使 witness_segment 净分不同的完整归属",
    )
    conflicts: list[dict] = Field(default_factory=list)
    verification: str | None = None
    verification_errors: list[str] = Field(default_factory=list)
    elapsed_ms: float


class SegmentAssignment(BaseModel):
    """见证名单中的逐段结果。"""

    segment_index: int
    duration_seconds: int
    players_on_court: list[int]
    confirmed_players: list[int]
    confirmed_included: bool
    target_all_on_court: bool
    team_net_points: int


class WitnessRoster(BaseModel):
    """一套完整逐段名单及其复核数据。"""

    label: str
    solver_status: str
    shared_seconds: int
    net_points: int
    meets_minimum: bool
    assignments: list[SegmentAssignment]
    player_total_seconds: dict[str, int]
    substitution_counts: list[int]


class NetBounds(BaseModel):
    """净胜分上下界及对应见证名单。"""

    min_net_points: int
    max_net_points: int
    min_witness: WitnessRoster
    max_witness: WitnessRoster


class CheckResponse(BaseModel):
    status: VerdictStatus
    status_label_zh: str
    target_lineup: list[int]
    minimum_seconds: int
    feasible: bool
    shared_seconds_min: int | None = None
    shared_seconds_max: int | None = None
    bounds_proven_optimal: bool = True
    net_bounds_all_feasible: NetBounds | None = None
    net_bounds_among_qualifying: NetBounds | None = None
    verification: str | None = None
    verification_errors: list[str] = Field(default_factory=list)
    conflicts: list[dict] = Field(default_factory=list)
    solver_runs: int = 0
    elapsed_ms: float


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str
