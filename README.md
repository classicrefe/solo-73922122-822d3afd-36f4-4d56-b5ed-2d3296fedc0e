# 篮球五人阵容达标核对 API

根据**官方逐人在场秒数**、**按换人切分的有序片段**（时长 / 已确认球员 / 相对前段换下人数 / 本队得失分），
用约束规划（Google OR-Tools CP-SAT）判定一套目标五人阵容能否达到最低同场上场时长，
并在达标名单中计算净胜分上下界，每个边界都附一套完整的逐段名单供独立复核。

## 判定语义

| 状态 | 含义 |
| --- | --- |
| `guaranteed` 必定达标 | 所有满足原始记录的名单，目标五人同场秒数都达到阈值 |
| `possible` 可能达标 | 存在达标名单，也存在不达标名单 |
| `impossible` 无法达标 | 原始记录可行，但没有任何名单达到阈值 |
| `infeasible_record` 原始记录冲突 | 约束无法同时成立；只返回冲突说明，不给达标结论 |

## 建模口径

- 变量 `x[t,p]`：片段 `t` 球员 `p` 是否在场（布尔）。
- 硬约束：
  1. 每段恰有 5 人，且包含该段全部已确认球员；
  2. 逐人累计在场秒数 `Σ duration[t]·x[t,p]` 等于官方秒数；
  3. 相邻片段变动人次 `Σ (x[t,p] XOR x[t-1,p])` 等于 `2 × 换下人数`
     （换下与换上人数相等，按双边人次计数；首段换下人数必须为 0）。
- 派生：
  - `z[t]`：目标五人是否在片段 `t` 同场；
  - 同场秒数 `S = Σ duration[t]·z[t]`；
  - 净胜分 `N = Σ team_net_points[t]·z[t]`（只累计目标五人同场的片段）。
- 对 `S`、`N` 分别最小化 / 最大化得到严格边界；达标名单的净胜分边界通过追加
  `S ≥ minimum_seconds` 约束后再次优化得到。
- 规模上限：12 名球员、16 个片段。

## 响应里的复核材料

每个净胜分边界都附带一套**完整逐段见证名单**，包含：

- 每段在场 5 人、已确认球员是否都在、目标五人是否同场；
- 逐人累计秒数（可直接与 `official_seconds` 对账）；
- 逐段实际换下人数（可与 `players_substituted` 对账）；
- 该名单的同场秒数与净胜分。

服务端在返回前会用**不依赖求解器的独立复核器**（纯 Python 重算全部约束）校验每套见证名单，
结果写在 `verification` / `verification_errors` 字段。

## 记分流水对账（`POST /api/v1/roster/reconcile`）

技术台的记分流水需要先归到换人片段，`team_net_points` 才能用于阵容净胜分分析。
对账接口接收**未填 `team_net_points`** 的 `/roster/check` 请求，外加：

| 字段 | 含义 |
| --- | --- |
| `scoring_entries` | 最多 120 笔有序记分；每笔给 `team_points` / `opponent_points`（各自增加的整数分）与 `candidate_segments`（候选片段号，可跳号、可多个） |
| `score_checkpoints` | 可选的片段末比分：`segment_number` 与该片段结束时双方本节累计得分 |

归属规则：

- **整笔只能归入一个候选片段**；已知同段的得分可以合记一笔，也可以拆成多笔；
- **片段号沿流水不得倒退**（因此同一片段的流水笔在流水中连续）；
- 每个核对点要求「截至该片段的双方累计」同时吻合；
- 片段净分 = 该段本队增分 − 该段对方增分。

判定结果：

| 状态 | 含义 | 返回 |
| --- | --- | --- |
| `resolved` | 每个片段净分在**所有合规归属下都相同** | 逐段 `resolved_team_net_points`、可直接提交 `/roster/check` 的 `request_body`、1 套见证归属 |
| `ambiguous` | 至少一个片段净分可能变化 | **全部**受影响片段 `affected_segments`；两套完整归属，使 `witness_segment` 的净分不同（供回看录像） |
| `score_conflict` | 记录无法同时成立 | 精确定位的 `conflicts`（如 `CHECKPOINT_NOT_ATTAINABLE`、`ENTRY_SEGMENT_ORDER_INFEASIBLE`） |

实现要点：`片段数 × 流水笔数` 的网格 DP 做前向 / 后向可达性，逐段枚举合法连续块。
主流程与 `app/verifier.py` 中**另一套独立实现**互相复核，
`verification_errors` 非空即代表两者不一致。

对账示例（`/docs` 中 `resolved_demo` / `ambiguous_demo` / `conflict_demo` 均可直接调用）：

```json
{
  "players": [1, 2, 3, 4, 5, 6, 7],
  "target_lineup": [2, 3, 5, 6, 7],
  "minimum_seconds": 100,
  "official_seconds": {"1": 300, "2": 600, "3": 600, "4": 300, "5": 300, "6": 300, "7": 600},
  "segments": [
    {"duration_seconds": 300, "confirmed_players": [2, 7], "players_substituted": 0},
    {"duration_seconds": 300, "confirmed_players": [3, 5, 7], "players_substituted": 2}
  ],
  "scoring_entries": [
    {"team_points": 4, "opponent_points": 0, "candidate_segments": [1]},
    {"team_points": 2, "opponent_points": 0, "candidate_segments": [1, 2]},
    {"team_points": 0, "opponent_points": 3, "candidate_segments": [2]}
  ],
  "score_checkpoints": [
    {"segment_number": 1, "team_cumulative_points": 4, "opponent_cumulative_points": 0}
  ]
}
```

返回 `resolved`，`resolved_team_net_points = [4, -1]`，`request_body` 即回填了
`team_net_points` 的 `/roster/check` 请求。三个示例预期结果：
`resolved_demo → resolved`、`ambiguous_demo → ambiguous`（受影响片段 `[1, 2]`）、
`conflict_demo → score_conflict`。

## 快速开始（Docker Compose）

```bash
docker compose up --build
```

- 交互式文档（含可直接调用的请求示例）：<http://localhost:8000/docs>
- 终场记分对账说明页：<http://localhost:8000/docs/reconcile>（随 /docs 一起可用）
- 健康检查：<http://localhost:8000/health>
- 接口：
  - `POST /api/v1/roster/check`：阵容达标核对；
  - `POST /api/v1/roster/reconcile`：记分流水对账（见下节）。

## 本地开发（无 Docker 时）

```bash
python3 -m venv --without-pip .venv
# 若系统没有 pip，可用 https://bootstrap.pypa.io/get-pip.py 引导
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload --port 8000
```

## 请求示例

```json
{
  "players": [1, 2, 3, 4, 5, 6, 7],
  "target_lineup": [2, 3, 5, 6, 7],
  "minimum_seconds": 100,
  "official_seconds": {"1": 300, "2": 600, "3": 600, "4": 300, "5": 300, "6": 300, "7": 600},
  "segments": [
    {"duration_seconds": 300, "confirmed_players": [2, 7], "players_substituted": 0, "team_net_points": 4},
    {"duration_seconds": 300, "confirmed_players": [3, 5, 7], "players_substituted": 2, "team_net_points": -3}
  ]
}
```

## 冲突诊断

记录无法同时成立时，`conflicts` 数组给出带错误码的精确定位，例如：

- `TOTAL_SECONDS_MISMATCH`：官方秒数总和不等于 5 人在场总人秒；
- `CONFIRMED_PLAYER_NOT_ENOUGH_SECONDS`：确认球员的官方秒数不够覆盖该片段；
- `SUBSTITUTION_CONFIRMED_CONFLICT`：确认球员的强制变动超过换人记录允许值；
- `NOT_ENOUGH_BENCH_PLAYERS`：换下人数超过板凳容量；
- `PLAYER_SECONDS_OUT_OF_REACH`：某球员官方秒数不在结构允许的可达区间内（附区间）；
- `JOINT_SECONDS_INFEASIBLE`：单人秒数都可达，但联合换人结构使它们无法同时满足；
- `STRUCTURAL_INFEASIBLE`：忽略秒数后五人 / 确认 / 换人结构本身就不成立。

对账接口（`/api/v1/roster/reconcile`）的冲突错误码：

- `ENTRY_HAS_NO_CANDIDATE_SEGMENT`：某笔记分的候选片段为空；
- `ENTRY_SEGMENT_ORDER_INFEASIBLE`：候选片段无法满足片段号沿流水不倒退；
- `CHECKPOINT_CUMULATIVE_DECREASE`：后一个核对点的累计小于前一个；
- `CHECKPOINT_NOT_ATTAINABLE`：核对点累计不等于流水的任何前缀（附最近前缀）；
- `CHECKPOINT_FINAL_TOTAL_MISMATCH`：最后一段核对点不等于全部记分合计；
- `CHECKPOINT_UNREACHABLE_AFTER_SEGMENT` / `SCORE_ATTRIBUTION_INFEASIBLE`：结构可达但仍无法同时成立。

## 测试

```bash
.venv/bin/pip install pytest httpx
.venv/bin/python -m pytest tests/ -q
```

测试策略：

- `tests/brute_force.py`：小规模实例的暴力枚举真值（独立于 CP-SAT 的第二套实现）；
- 250 个随机实例（含可行 / 扰动冲突变体）逐项对照状态、同场秒数与净胜分边界；
- 四种判定状态的冻结夹具、阈值边界、422 结构校验；
- 12 人 × 16 段规模上限的性能与正确性（秒级完成，边界均证明最优）；
- 独立复核器对各类人为篡改的检出测试；
- 记分对账：3 个 OpenAPI 示例均可调用且状态符合预期、回填请求体可直接提交
  `/roster/check`、400 个随机实例与「枚举全部归属」的暴力真值逐项对照
  （状态 / 受影响片段 / 逐段净分 / 两套见证归属），以及 120 笔 × 16 段规模性能。
