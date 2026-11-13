# 星塔规则与实机分析

游戏规格入口为[潜能状态空间](../Background/StateSpace.md)与[音符状态空间](../Background/NoteStateSpace.md)；[潜能观察清单](../Background/StateSpace_Observation_Targets.md)和[音符观察报告](../Background/NoteStateSpace_Observation_Targets.md)分别维护实测支持、工作程序与未定参数。

- `data/`：[音符／协奏／秘纹与阶段供给](data/NoteRules.json)、[塔与事件配置](data/TowerRuleTables.json)、[协议字段](data/NoteProtocol.json)、[效果参数](data/stella_note_effects_verified.json)，以及必要的聚合研究结果。静态来源保留标识、版本和SHA256。
- `reports/`：[音符总库分析](reports/NoteRealGameValidation.md)说明来源、分母及候选比较；[秘纹目录对照](reports/DiscCatalogCoverage.md)区分104件历史线上核对与106件当前静态目录；[开局预设预测评测](reports/PresetForecast.md)对比小模型与10万局模拟的终局分布。
- `analysis/`：[导入原生对局](analysis/IMPORT_RUNS.md)将指定文件合入自己的 SQLite 库；[审核音符观测](analysis/NOTE_QUALITY.md)按实际动作统计重读、冲突与已知库存。已有 `realgame_*` 脚本是早期分析工具，其样本口径见下表。

统一实机数据位于 `Background/RealGameData/`，读取方法见[数据说明](../Background/RealGameData/README.txt)，最新全库计数见[SUMMARY.json](../Background/RealGameData/SUMMARY.json)。记录数、采集段和物理局分别统计；统计单位采用实际来源区间、菜单或货架generation，重读不增加独立样本数。

| 材料 | 时间与范围 | 使用方式 |
|---|---|---|
| [音符聚合研究结果](data/note_database_summary.json)与[音符总库分析](reports/NoteRealGameValidation.md) | 2026-11-13分析快照，观测截至11-12；649403条records、652个物理局，其中97局含音符 | 机制拟合与观察结论；1155个有效战斗、120个事件、3489架完整货架 |
| `realgame_*` JSON及[NoteValidationCapturePlan.json](data/NoteValidationCapturePlan.json) | 2026-11-08早期快照；401040条记录、391个participant/session/run组、1809架货架 | 保留该批统计与字段规划；其中完整音符库存／独立战斗变化为0不代表后续总库 |
| `RealGameData/SUMMARY.json`与[note_quality.py](analysis/note_quality.py) | 当前已合并总库 | 当前规模及质量盘点，不自动改写上面两批已完成的机制分析 |

各批结果使用自己的身份归并和字段资格。新增记录及修正先进入总库，只有相关分析实际重算后才更新报告数字；不把数据库规模变化直接当作机制试验数增加。潜能观察清单中的2026-11-09新增66局、等级／NPC及E6比较也保留原分析口径。

本目录公开规则、静态数值及聚合研究结论。客户端载荷、还原源码、逐局研究中间材料、联合训练器代码与权重不随此目录发布。
