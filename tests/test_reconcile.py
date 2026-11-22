"""记分流水对账接口测试：示例、边界、与独立暴力枚举逐项对照。"""
from __future__ import annotations

import itertools
import json
import random
import time

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.verifier import attribution_options, verify_attribution

client = TestClient(app)


BASE = {
    "players": [1, 2, 3, 4, 5, 6, 7],
    "target_lineup": [2, 3, 5, 6, 7],
    "minimum_seconds": 100,
    "official_seconds": {
        "1": 300, "2": 600, "3": 600, "4": 300, "5": 300, "6": 300, "7": 600,
    },
    "segments": [
        {"duration_seconds": 300, "confirmed_players": [2, 7], "players_substituted": 0},
        {"duration_seconds": 300, "confirmed_players": [3, 5, 7], "players_substituted": 2},
    ],
}


def reconcile(**overrides) -> dict:
    payload = json.loads(json.dumps(BASE))
    payload.update(overrides)
    resp = client.post("/api/v1/roster/reconcile", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


def brute_force(req: dict) -> dict:
    """枚举全部「片段号不倒退 + 整笔落一个候选片段」的归属，独立于服务实现。"""
    T = len(req["segments"])
    entries = req["scoring_entries"]
    cps = {c["segment_number"]: c for c in req["score_checkpoints"]}

    feasible_assignments = []
    for combo in itertools.product(
        *[e["candidate_segments"] for e in entries]
    ):
        if any(b < a for a, b in zip(combo, combo[1:])):
            continue
        ok = True
        for s, cp in cps.items():
            team = sum(e["team_points"] for e, seg in zip(entries, combo) if seg <= s)
            opp = sum(
                e["opponent_points"] for e, seg in zip(entries, combo) if seg <= s
            )
            if (team, opp) != (
                cp["team_cumulative_points"],
                cp["opponent_cumulative_points"],
            ):
                ok = False
                break
        if not ok:
            continue
        seg_team = [0] * T
        seg_opp = [0] * T
        for e, seg in zip(entries, combo):
            seg_team[seg - 1] += e["team_points"]
            seg_opp[seg - 1] += e["opponent_points"]
        feasible_assignments.append(
            ([seg_team[t] - seg_opp[t] for t in range(T)], list(combo))
        )

    if not feasible_assignments:
        return {"feasible": False}
    nets_per_segment = [set() for _ in range(T)]
    for nets, _ in feasible_assignments:
        for t, n in enumerate(nets):
            nets_per_segment[t].add(n)
    return {
        "feasible": True,
        "assignments": feasible_assignments,
        "possible_nets": nets_per_segment,
    }


def assert_witness(req: dict, witness: dict) -> None:
    assert verify_attribution(req, [a["segment_number"] for a in witness["entry_assignments"]]) == []
    assert len(witness["entry_assignments"]) == len(req["scoring_entries"])
    for a in witness["entry_assignments"]:
        assert a["net_points"] == a["team_points"] - a["opponent_points"]
    for t, net in enumerate(witness["segment_net_points"]):
        assert net == (
            witness["segment_team_points"][t] - witness["segment_opponent_points"][t]
        )
    for cp in witness["checkpoints"]:
        assert cp["matches"] is True


# --------------------------------------------------------------------------- #
# 三个 OpenAPI 示例都是可调用的契约
# --------------------------------------------------------------------------- #
def test_openapi_examples_are_callable() -> None:
    spec = client.get("/openapi.json").json()
    content = spec["paths"]["/api/v1/roster/reconcile"]["post"]["requestBody"][
        "content"
    ]["application/json"]
    assert set(content["examples"]) == {
        "resolved_demo",
        "ambiguous_demo",
        "conflict_demo",
    }
    got = {}
    for name, ex in content["examples"].items():
        resp = client.post("/api/v1/roster/reconcile", json=ex["value"])
        assert resp.status_code == 200, name
        body = resp.json()
        got[name] = body["status"]
        assert body["verification_errors"] == []
    assert got == {
        "resolved_demo": "resolved",
        "ambiguous_demo": "ambiguous",
        "conflict_demo": "score_conflict",
    }


def test_demo_page_serves_callable_examples() -> None:
    resp = client.get("/docs/reconcile")
    assert resp.status_code == 200
    text = resp.text
    assert "/api/v1/roster/reconcile" in text
    for name in ("resolved_demo", "ambiguous_demo", "conflict_demo"):
        assert name in text
    assert client.get("/docs").status_code == 200


def test_resolved_demo_body_is_accepted_by_roster_check() -> None:
    spec = client.get("/openapi.json").json()
    ex = spec["paths"]["/api/v1/roster/reconcile"]["post"]["requestBody"][
        "content"
    ]["application/json"]["examples"]["resolved_demo"]["value"]
    body = client.post("/api/v1/roster/reconcile", json=ex).json()
    assert body["resolved_team_net_points"] == [4, -1]
    assert [s["team_net_points"] for s in body["request_body"]["segments"]] == [4, -1]
    assert "team_net_points" not in ex["segments"][0]
    check = client.post("/api/v1/roster/check", json=body["request_body"])
    assert check.status_code == 200
    assert check.json()["feasible"] is True


def test_ambiguous_lists_all_affected_and_two_complete_attributions() -> None:
    body = reconcile(
        scoring_entries=[
            {"team_points": 2, "opponent_points": 0, "candidate_segments": [1, 2]},
            {"team_points": 0, "opponent_points": 1, "candidate_segments": [1, 2]},
        ]
    )
    assert body["status"] == "ambiguous"
    assert body["affected_segments"] == [1, 2]
    assert body["resolved_team_net_points"] is None
    assert body["request_body"] is None
    witness_segment = body["witness_segment"]
    a, b = body["witnesses"]
    assert len(a["entry_assignments"]) == len(b["entry_assignments"]) == 2
    assert (
        a["segment_net_points"][witness_segment - 1]
        != b["segment_net_points"][witness_segment - 1]
    )

    req = json.loads(json.dumps(BASE))
    req["scoring_entries"] = [
        {"team_points": 2, "opponent_points": 0, "candidate_segments": [1, 2]},
        {"team_points": 0, "opponent_points": 1, "candidate_segments": [1, 2]},
    ]
    for w in body["witnesses"]:
        assert_witness(req, w)


def test_empty_stream_resolves_to_zero_net() -> None:
    body = reconcile(scoring_entries=[])
    assert body["status"] == "resolved"
    assert body["resolved_team_net_points"] == [0, 0]
    assert body["witnesses"][0]["entry_assignments"] == []


def test_checkpoints_can_collapse_ambiguity() -> None:
    entries = [
        {"team_points": 2, "opponent_points": 0, "candidate_segments": [1, 2]},
        {"team_points": 0, "opponent_points": 1, "candidate_segments": [1, 2]},
    ]
    assert reconcile(scoring_entries=entries)["status"] == "ambiguous"
    body = reconcile(
        scoring_entries=entries,
        score_checkpoints=[
            {"segment_number": 1, "team_cumulative_points": 2,
             "opponent_cumulative_points": 0},
            {"segment_number": 2, "team_cumulative_points": 2,
             "opponent_cumulative_points": 1},
        ],
    )
    assert body["status"] == "resolved"
    assert body["resolved_team_net_points"] == [2, -1]


def test_scores_of_same_segment_may_be_merged_into_one_entry() -> None:
    # 两笔都只能进片段 1，与一笔 (3,0) 等价；整笔只能归入一个片段。
    merged = reconcile(
        scoring_entries=[
            {"team_points": 3, "opponent_points": 0, "candidate_segments": [1]},
        ]
    )
    split = reconcile(
        scoring_entries=[
            {"team_points": 2, "opponent_points": 0, "candidate_segments": [1]},
            {"team_points": 1, "opponent_points": 0, "candidate_segments": [1]},
        ]
    )
    assert merged["resolved_team_net_points"] == split["resolved_team_net_points"] == [3, 0]
    assert merged["witnesses"][0]["segment_team_points"] == [3, 0]
    assert split["witnesses"][0]["segment_team_points"] == [3, 0]


# --------------------------------------------------------------------------- #
# 冲突诊断
# --------------------------------------------------------------------------- #
def test_conflict_checkpoint_not_attainable() -> None:
    body = reconcile(
        scoring_entries=[
            {"team_points": 4, "opponent_points": 0, "candidate_segments": [1]},
            {"team_points": 0, "opponent_points": 3, "candidate_segments": [2]},
        ],
        score_checkpoints=[
            {"segment_number": 1, "team_cumulative_points": 5,
             "opponent_cumulative_points": 0},
        ],
    )
    assert body["status"] == "score_conflict"
    assert body["conflicts"][0]["code"] == "CHECKPOINT_NOT_ATTAINABLE"
    assert body["verification_errors"] == []
    assert body["resolved_team_net_points"] is None


def test_conflict_final_checkpoint_total_mismatch() -> None:
    resp = client.post(
        "/api/v1/roster/reconcile",
        json={
            **json.loads(json.dumps(BASE)),
            "scoring_entries": [
                {"team_points": 2, "opponent_points": 0, "candidate_segments": [1, 2]}
            ],
            "score_checkpoints": [
                {"segment_number": 2, "team_cumulative_points": 3,
                 "opponent_cumulative_points": 0}
            ],
        },
    )
    body = resp.json()
    assert body["status"] == "score_conflict"
    assert body["conflicts"][0]["code"] == "CHECKPOINT_FINAL_TOTAL_MISMATCH"


def test_conflict_segment_order() -> None:
    body = reconcile(
        scoring_entries=[
            {"team_points": 1, "opponent_points": 0, "candidate_segments": [2]},
            {"team_points": 0, "opponent_points": 1, "candidate_segments": [1]},
        ]
    )
    assert body["status"] == "score_conflict"
    assert body["conflicts"][0]["code"] == "ENTRY_SEGMENT_ORDER_INFEASIBLE"


def test_conflict_entry_without_candidate() -> None:
    body = reconcile(
        scoring_entries=[
            {"team_points": 2, "opponent_points": 0, "candidate_segments": []},
        ]
    )
    assert body["status"] == "score_conflict"
    assert body["conflicts"][0]["code"] == "ENTRY_HAS_NO_CANDIDATE_SEGMENT"


def test_conflict_cumulative_decrease_between_checkpoints() -> None:
    body = reconcile(
        scoring_entries=[
            {"team_points": 2, "opponent_points": 0, "candidate_segments": [1, 2]},
        ],
        score_checkpoints=[
            {"segment_number": 1, "team_cumulative_points": 2,
             "opponent_cumulative_points": 0},
            {"segment_number": 2, "team_cumulative_points": 1,
             "opponent_cumulative_points": 0},
        ],
    )
    assert body["status"] == "score_conflict"
    assert body["conflicts"][0]["code"] == "CHECKPOINT_CUMULATIVE_DECREASE"


# --------------------------------------------------------------------------- #
# 422 结构校验
# --------------------------------------------------------------------------- #
def test_validation_rejects_filled_team_net_points() -> None:
    payload = json.loads(json.dumps(BASE))
    payload["segments"][0]["team_net_points"] = 3
    payload["scoring_entries"] = []
    assert client.post("/api/v1/roster/reconcile", json=payload).status_code == 422


def test_validation_rejects_unknown_entry_field() -> None:
    payload = json.loads(json.dumps(BASE))
    payload["scoring_entries"] = [
        {"team_points": 1, "opponent_points": 0,
         "candidate_segments": [1], "quarter": 1}
    ]
    assert client.post("/api/v1/roster/reconcile", json=payload).status_code == 422


def test_validation_rejects_too_many_entries_and_bad_numbers() -> None:
    payload = json.loads(json.dumps(BASE))
    payload["scoring_entries"] = [
        {"team_points": 0, "opponent_points": 0, "candidate_segments": [1]}
    ] * 121
    assert client.post("/api/v1/roster/reconcile", json=payload).status_code == 422

    payload["scoring_entries"] = []
    payload["score_checkpoints"] = [
        {"segment_number": 3, "team_cumulative_points": 0,
         "opponent_cumulative_points": 0}
    ]
    assert client.post("/api/v1/roster/reconcile", json=payload).status_code == 422

    payload["score_checkpoints"] = [
        {"segment_number": 1, "team_cumulative_points": 0,
         "opponent_cumulative_points": 0},
        {"segment_number": 1, "team_cumulative_points": 0,
         "opponent_cumulative_points": 0},
    ]
    assert client.post("/api/v1/roster/reconcile", json=payload).status_code == 422


# --------------------------------------------------------------------------- #
# 随机实例与暴力枚举逐项对照
# --------------------------------------------------------------------------- #
def random_request(rng: random.Random) -> dict:
    T = rng.randint(1, 4)
    payload = {
        "players": [1, 2, 3, 4, 5],
        "target_lineup": [1, 2, 3, 4, 5],
        "minimum_seconds": 0,
        # 5 人打满全部片段：阵容记录本身恒可行，便于回填后直接提交 /roster/check
        "official_seconds": {str(p): 100 * T for p in range(1, 6)},
        "segments": [
            {"duration_seconds": 100, "confirmed_players": [], "players_substituted": 0}
            for _ in range(T)
        ],
    }
    n = rng.randint(0, 5)
    # 生成「不倒退」的真实片段序列，再据此造候选（保留原地 + 随机加宽/收窄）
    truth = sorted(rng.randint(1, T) for _ in range(n))
    entries = []
    for seg in truth:
        candidates = {seg}
        if rng.random() < 0.5:
            candidates.update(rng.sample(range(1, T + 1), rng.randint(1, T)))
        entries.append(
            {
                "team_points": rng.randint(0, 3),
                "opponent_points": rng.randint(0, 3),
                "candidate_segments": sorted(candidates),
            }
        )
    checkpoints = []
    if rng.random() < 0.5:
        picked = sorted(rng.sample(range(1, T + 1), rng.randint(1, T)))
        for seg in picked:
            shift = rng.choice([0, 0, 0, 1, -1])
            checkpoints.append(
                {
                    "segment_number": seg,
                    "team_cumulative_points": max(
                        0,
                        sum(e["team_points"] for e, s in zip(entries, truth) if s <= seg)
                        + shift,
                    ),
                    "opponent_cumulative_points": max(
                        0,
                        sum(
                            e["opponent_points"]
                            for e, s in zip(entries, truth)
                            if s <= seg
                        )
                        + rng.choice([0, 0, 1]),
                    ),
                }
            )
        if len({c["segment_number"] for c in checkpoints}) < len(checkpoints):
            checkpoints = []
    payload["scoring_entries"] = entries
    payload["score_checkpoints"] = sorted(checkpoints, key=lambda c: c["segment_number"])
    return payload


def test_random_against_brute_force() -> None:
    rng = random.Random(424242)
    counts = {"resolved": 0, "ambiguous": 0, "score_conflict": 0}
    for _ in range(400):
        req = random_request(rng)
        resp = client.post("/api/v1/roster/reconcile", json=req)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        truth = brute_force(req)
        counts[body["status"]] += 1

        # 独立复核器与暴力枚举也必须一致
        if truth["feasible"]:
            assert body["status"] in ("resolved", "ambiguous")
            options = attribution_options(req)
            assert options["feasible"] is True
            assert options["possible_nets"] == truth["possible_nets"]
        else:
            assert body["status"] == "score_conflict"
            assert attribution_options(req)["feasible"] is False
            assert body["verification"] is not None

        if body["status"] == "score_conflict":
            assert body["conflicts"]
            assert body["verification_errors"] == []
            continue

        expected_affected = [
            t + 1 for t, s in enumerate(truth["possible_nets"]) if len(s) > 1
        ]
        assert body["affected_segments"] == expected_affected

        if body["status"] == "resolved":
            nets = [next(iter(s)) for s in truth["possible_nets"]]
            assert body["resolved_team_net_points"] == nets
            attrs = body["witnesses"][0]["entry_assignments"]
            assert verify_attribution(req, [a["segment_number"] for a in attrs]) == []
            check = client.post("/api/v1/roster/check", json=body["request_body"])
            assert check.status_code == 200
            assert check.json()["feasible"] is True
        else:
            assert body["resolved_team_net_points"] is None
            assert body["request_body"] is None
            assert body["witness_segment"] in expected_affected
            assert len(body["witnesses"]) == 2
            nets = [w["segment_net_points"] for w in body["witnesses"]]
            for w in body["witnesses"]:
                attrs = [a["segment_number"] for a in w["entry_assignments"]]
                assert verify_attribution(req, attrs) == []
            ws = body["witness_segment"] - 1
            assert nets[0][ws] != nets[1][ws]
            # 两套见证的逐段净分都必须确实落在暴力枚举的可能集合里
            for plan_nets in nets:
                for t, n in enumerate(plan_nets):
                    assert n in truth["possible_nets"][t]
            # 列出全部受影响片段，而不是只给一个
            for t in expected_affected:
                assert t in body["affected_segments"]
        assert body["verification_errors"] == []

    # 三种状态都应被随机样本覆盖
    assert all(v > 10 for v in counts.values()), counts


def test_max_scale_performance() -> None:
    rng = random.Random(9)
    T, N = 16, 120
    truth = sorted(rng.randint(1, T) for _ in range(N))
    entries = [
        {
            "team_points": rng.randint(0, 3),
            "opponent_points": rng.randint(0, 3),
            "candidate_segments": sorted(
                {seg} | set(rng.sample(range(1, T + 1), rng.randint(0, T)))
            ),
        }
        for seg in truth
    ]
    checkpoints = []
    for seg in (4, 8, 12, 16):
        checkpoints.append(
            {
                "segment_number": seg,
                "team_cumulative_points": sum(
                    e["team_points"] for e, s in zip(entries, truth) if s <= seg
                ),
                "opponent_cumulative_points": sum(
                    e["opponent_points"] for e, s in zip(entries, truth) if s <= seg
                ),
            }
        )
    payload = {
        **json.loads(json.dumps(BASE)),
        "segments": [
            {"duration_seconds": 100, "confirmed_players": [], "players_substituted": 0}
            for _ in range(T)
        ],
        "official_seconds": {str(p): 0 for p in BASE["players"]},
        "scoring_entries": entries,
        "score_checkpoints": checkpoints,
    }
    started = time.perf_counter()
    resp = client.post("/api/v1/roster/reconcile", json=payload)
    elapsed = time.perf_counter() - started
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] in ("resolved", "ambiguous")
    assert body["verification_errors"] == []
    assert elapsed < 5.0, f"120 笔 × 16 段耗时 {elapsed:.2f}s"
