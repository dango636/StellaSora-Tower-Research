# 模型侧音符数据导入（尚未绑定 Maa 原生数据库）

2026-11-14检查时，原插件仓库已经归档，协作者[11月10日的说明](https://github.com/dango636/MaaStellaSora-NeuralNetwork/pull/1#issuecomment-6770445115)表示采集及数据库已进入Maa原生分支。当前公开研究仓库与实机release仍是11月8日的版本；本工具尚未取得该原生数据库结构和新导出样本。**本工具不是已完成的原生数据库适配，也不启动训练。** SQLite与JSONL是目前实现的接收格式，不代表已经确认上游选用哪一种数据库。

交付内容：只读结构检查器、显式映射导入器、保守完整性检查、重复／冲突处理、合成演示及测试。没有改动Maa、现有训练器、权重或现有总数据。只需Python标准库。

## 先检查结构

从仓库根目录运行，输出文件必须尚不存在：

```powershell
python Research/analysis/note_capture_import.py inspect /path/to/frozen-export.sqlite --format sqlite --output /path/to/profile.json
python Research/analysis/note_capture_import.py inspect /path/to/export.jsonl --format jsonl --output /path/to/profile.json
```

SQLite只读取表／视图名称、列名称／声明类型／约束；不会输出建表SQL中的默认字面量或原始行。JSONL默认采样前1000行，记录字段路径与类型，明确是否只是采样；不会把采样缺失宣称为整包没有音符数据。数字音符ID键保留，其他动态非标识键用通配符归并。完整源文件SHA256与结构摘要分别记录。

也可以直接检查旧公开ZIP，不解压原始玩家记录：

```powershell
python Research/analysis/note_capture_import.py inspect /path/to/real-game-data.zip --format jsonl_zip --member Background/RealGameData/data/observations.jsonl --output /path/to/old-package-profile.json
```

数据库须为静态导出，非空WAL／journal旁文件会导致拒绝。请使用数据库提供方的SQLite backup／导出流程生成一致快照，不在正在写入的数据库上试验。连接为只读、immutable且禁用扩展；显式数据查询还受SQLite authorizer限制，拒绝写入、ATTACH、PRAGMA和加载扩展。导入前后重新核对源SHA256；源发生变化就不发布结果。

## 原生格式绑定要等真实结构

`note_capture_mapping_template.json`有意保持 `binding.status=unbound`，没有猜测原生表名、列名、联结关系或成功标记。拿到原生分支的具体提交、结构和匿名快照后再填写：

- `source_format`：`sqlite`、`jsonl`或`jsonl_zip`；ZIP需要明确的member路径。
- `namespace`：稳定的数据集命名空间。同一采集源后续导出继续用同一命名空间。
- `binding`：`reviewed_source_binding`、结构SHA256、生产者提交与对应字段语义的核对证据。工具核对结构摘要和声明完整性，不能独立证明该语义声明真实；报告明确保留这一限制。
- `query`：SQLite的显式只读SELECT，可按实际结构联结观测、候选、实际动作与结算；重名输出列会拒绝。JSONL不需要query。
- `identity.run`／`identity.action`：来源路径的数组。run通常需要参与者、会话、对局的组合键，避免不同玩家使用同一run编号发生碰撞。输出使用命名空间内的哈希标识；不输出原始身份键。
- `json_columns`：明确列出的SQLite JSON文本列。不会把所有文本列都当JSON，也不会执行源数据内容。
- `fields`：规范字段到来源路径的映射；路径是字符串／数组索引组成的列表。只有显式声明的`bool01`和`enum`转换，没有隐式真假转换、OCR解析或字段别名猜测。

原生字段需要覆盖以下信息：

| 内容 | 规范字段 |
| --- | --- |
| 动作及顺序 | 组合run/action身份、sequence、event_type、已核对的option_id |
| 游戏结果确认 | operation_confirmed、response_linked、confirmation_source |
| 来源和识别质量 | source_complete、recognition_status、invalid_fields、intervening_note_actions |
| 前后库存 | before／after.notes、notes_complete；90011到90023各一项，未知保持null |
| 协议增减（如有） | note_infos的Tid／有符号Qty；缺失保持null，不编空数组 |
| 条件 | 客户端版本、区服、塔、楼层、主辅秘纹ID和明确阶段 |
| 筛选许可 | training_eligible、probability_eligible、sequence_complete、sequence_inference_eligible |

缺字段不会默认成true、0或空数组。显示位置不能猜成静态事件ID；Maa_success不能映射为游戏结果已确认。成功标记等证明字段不能直接设置常量。真实字段含义不一致时，应先与生产者核对，不能只改映射让样本通过。

## 导出与拒绝口径

```powershell
python Research/analysis/note_capture_import.py import /path/to/frozen-export.sqlite --mapping /path/to/reviewed-mapping.json --output-directory /path/to/new-output-directory
```

结果目录包含：

- `candidates.jsonl`：通过当前完整性检查的唯一动作，只保留规范字段和导入来源摘要。通过检查仍不等于真实随机核已校准或可以训练。
- `rejections.jsonl`：哈希动作标识、源行数与组级拒绝状态，不包含原始行、OCR文本或身份值。
- `report.json`：源结构和摘要、质量计数、逐原因计数，以及明确的 `training_approved=false`。

完整13种库存只接受实读非负整数；null不补零，不从其他局或上一次读数回填。所有许可和确认标记必须明确为true，识别状态为valid、失效字段为空；缺失也会拒绝。若存在协议Qty，会检查带符号增减与前后库存一致。103／104兑换还校验总扣除5／10、无正增量。

同动作相同重读只计一次；只要同动作有冲突或任一无效读数，该动作整组排除，不先删除坏读数再保留有利结果。同一对局同一序号对应不同动作也整组排除。分组结果按首次源观察顺序输出，保留映射得到的动作序号；不据此重建缺失历史或构造新的奖励次数。需要重新观测才能解除的冲突应在生产者一侧明确记录，不能在导入时猜选其中一条。

输出使用临时磁盘分组，不把整库常驻内存；完整读取、分组和源检查成功后才发布目录。拒绝覆盖已有结果。源JSON畸形、重复键、非有限数值、SQL写入或查询失败不会留下看似成功的半批结果。

## 合成演示（不是真实样本）

所有表名和记录仅用于测试接口，不冒充原生格式。选择一个不存在的本地演示目录：

```powershell
python Research/analysis/note_capture_demo.py --output-directory /path/to/demo
python Research/analysis/note_capture_import.py import /path/to/demo/SYNTHETIC.sqlite --mapping /path/to/demo/SYNTHETIC.mapping.json --output-directory /path/to/demo-result --allow-test-profile
python -m unittest discover -s Research/analysis/tests -p 'test_note_capture_import.py' -v
```

演示含6次观察：一条动作的两次相同重读、一条动作的一好一坏冲突，以及缺库存／未确认操作。预期只有1个合成动作通过结构检查，真实动作数始终0；输出合成行还强制 `training_eligible=false`，防止误入旧训练筛选器。没有`--allow-test-profile`会拒绝测试映射。

## 接下来的实际工作

当前缺少的是原生分支可定位的提交、数据库结构与一份可核对的匿名导出，不需要另建采集器或继续维护外置插件。拿到后绑定字段，验证同动作前后库存与游戏结果，再按真实对局、客户端／秘纹条件分组统计战斗／扣除机制。只有证据和规则齐备，才讨论联合策略的普通选牌、买音符、强化和事件训练。本导入器没有连接训练入口。
