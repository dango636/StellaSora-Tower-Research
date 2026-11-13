# 原生音符区间只读审核

`note_quality.py` 仅依赖 Python 标准库，直接审核 Maa 原生 SQLite，或含 `research.json` 与 `manifest.json` 的原生单局 ZIP。只输出汇总数和固定原因码，不输出玩家标识、动作标识、路径、OCR 原文或库存明细，不生成样本或训练标签。

从仓库根目录运行：

```powershell
python Research/analysis/note_quality.py /path/to/research.sqlite3
python Research/analysis/note_quality.py /path/to/ended-run.zip
python Research/analysis/note_quality.py /path/to/research.sqlite3 --run-id <native-run-id>
```

SQLite 以 `mode=ro` 打开，在同一读事务中使用现有 `record_quality`、`note_deltas` 视图。ZIP 按公开的 [research_schema.sql](../../Background/RealGameData/research_schema.sql) 将八张原生表载入内存，再执行同一审核；不解压至磁盘、不调用 Maa 或训练器。ZIP 的格式、版本和单局标识须与 manifest 一致。未结束的 `open` 局只计入跳过数；`completed` 与 `interrupted` 局均可提供已经独立确认的局部区间，不要求整局无瑕疵。

## 统计口径

- 以原生 `(run_id, operation_id)` 为动作单位。相同动作重读仅计一次；任何一份读数不合格，或组内库存、归因、资格等证据冲突，整组排除。记录序号是观测顺序，不额外假定它是动作序号。
- 保留原生视图的独立来源、分析、概率资格和有效质量判定，包括旧格式的原生缺省语义。综合区间保留为综合区间，不能拆成单个事件或战斗的独立样本。
- 普通战斗只核对七种通用音符及 `active_melodies` 指定的本塔两种属性音符；塔外未知项不补零，也不因塔外读数补全而把同一战斗变成冲突。核对实存非负整数、带符号差值和范围内总量。
- 其他事件、购买和兑换使用原生声明范围。`local_statistical_actions.exchange` 只表示该范围内的区间通过检查，不能用来拟合完整兑换随机核。
- `exchange_strict_13_inventory_checks` 是上述合格兑换的严格研究子集：原生范围覆盖全部13种、前后库存均已知、配置记录明确为随机扣除且声明数量、没有正增量、总扣除与声明一致。条件来自原生 `note_effect`，不猜静态事件 ID，也不把模型预期当成游戏执行成功证明。
- `exchange_hidden_supply_unknown_groups` 保留塔外库存未知的兑换组数；`exchange_strict_subset_reason_counts` 说明不能进入13维子集的库存或规则原因。子集仍须同时满足前述原生资格；局部排除原因另见 `exclusion_reason_counts`。各原因计数可重叠。

这是**局部观测统计审核**。已知库存可以带有原生的历史读数或规则来源，不能将结果说成13项均在同一帧新做了 OCR。脚本不独立证明生产者的归因、兑换随机规律、完整历史、4960预设范围或训练资格，始终输出 `training_approved=false`。

## 极小合成示例

以下示例完全合成，只在内存创建原生表，不包含真实对局、身份或路径。它包含：同一战斗的两次相同读数、一个存在冲突的战斗、一份本塔未知读数、一份13维兑换、一份塔外未知兑换。PowerShell 中从仓库根目录运行：

```powershell
@'
import copy, json, sqlite3, sys
from pathlib import Path
sys.path.insert(0, str(Path("Research/analysis").resolve()))
from note_quality import audit, COMMON, MELODIES, SCHEMA

db = sqlite3.connect(":memory:")
db.executescript(SCHEMA.read_text(encoding="utf-8"))
db.execute("""INSERT INTO runs
 (run_id,lifecycle_status,termination_basis,full_run_start_observed,
  completion_observed,quality_status,runtime_versions_json,model_versions_json,
  config_json,observation_kind)
 VALUES ('SYNTHETIC','completed','synthetic',1,1,'valid','{}','{}','{}','synthetic')""")
scope = list(COMMON) + ["melody_of_lux", "melody_of_terra"]
before = {key: 10 if key in scope else None for key in MELODIES}
after = dict(before); after[scope[0]] += 2
battle = dict(operation_id="SYNTHETIC-battle",source_type="battle",floor=1,
 floor_complete=True,complete=True,scope_melodies=scope,
 active_melodies=scope[-2:],before=before,after=after,
 delta={k: after[k]-before[k] if before[k] is not None else None for k in MELODIES},
 total=2,probability_eligible=True)
conflict = copy.deepcopy(battle); conflict["operation_id"] = "SYNTHETIC-conflict"
changed = copy.deepcopy(conflict)
changed["after"][scope[0]] += 1; changed["delta"][scope[0]] += 1; changed["total"] += 1
unknown = copy.deepcopy(battle); unknown["operation_id"] = "SYNTHETIC-unknown"
unknown["after"][scope[0]] = None; unknown["delta"][scope[0]] = None
exchange = copy.deepcopy(battle)
exchange.update(operation_id="SYNTHETIC-exchange",source_type="event",floor=2,
 scope_melodies=list(MELODIES),note_effect=dict(operation="spend",selection="random",count=5),
 before=dict.fromkeys(MELODIES,10),after=dict.fromkeys(MELODIES,10),
 delta=dict.fromkeys(MELODIES,0),total=-5)
exchange["after"][scope[0]] -= 5; exchange["delta"][scope[0]] = -5
hidden = copy.deepcopy(exchange); hidden["operation_id"] = "SYNTHETIC-hidden"
hidden["scope_melodies"] = scope
for key in set(MELODIES)-set(scope):
 for side in ("before","after","delta"): hidden[side][key] = None
for i, payload in enumerate((battle,battle,conflict,changed,unknown,exchange,hidden),1):
 db.execute("""INSERT INTO records
 (record_id,original_record_id,run_id,origin,sequence,event_type,basis,quality_status,payload_json)
 VALUES (?,?,'SYNTHETIC','run',?,'note_delta','inventory_difference_derivation','valid',?)""",
 (str(i),str(i),i,json.dumps(payload)))
print(json.dumps(audit(db),indent=2))
db.close()
'@ | python -
```

预期：7次观测、5个动作组、2次非独立重读、1个冲突组；合格局部统计为战斗1、兑换2，排除2组；13维兑换子集1、隐藏供给未知兑换1，`training_approved=false`。示例中的许可仅为合成输入，不能作为真实采集已获资格的证据。

## 借鉴与署名

感谢北辰（[beichen24a1](https://github.com/beichen24a1)）在 [PR #1](https://github.com/dango636/StellaSora-Tower-Research/pull/1) 中提出的未知不补零、按动作去重、冲突或无效读数整组排除、完整性检查不等于训练许可等校验思想。本脚本针对已有原生 SQL 表和视图另行实现，未合并该 PR，未复制其通用映射导入器或代码；也不把贡献者的实现称为本项目原创。
