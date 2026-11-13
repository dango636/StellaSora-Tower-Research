# 星塔规则与实机分析

游戏规则入口为[潜能状态空间](../Background/StateSpace.md)、[音符状态空间](../Background/NoteStateSpace.md)和[观察目标](../Background/StateSpace_Observation_Targets.md)。

- `data/`：音符、协奏、秘纹与供给目录，相关塔／事件配置，协议字段，效果参数，以及实机派生统计和补录要求。客户端资料保留来源标识与摘要，不附客户端载荷或底层实现文件。
- `reports/`：[音符实机统计](reports/NoteRealGameValidation.md)与[秘纹目录对照](reports/DiscCatalogCoverage.md)。
- `analysis/`：本次实机研究使用的7个分析／生成脚本，仅商店统计需要NumPy。

模型侧新采集数据的结构检查与显式映射导入见 [NOTE_CAPTURE_IMPORT.md](analysis/NOTE_CAPTURE_IMPORT.md)。该工具尚未绑定Maa原生数据库格式，不改写总数据、不启动训练。

实机统计基于2026-10-21至2026-11-08的401040条记录。输入使用唯一总数据路径`Background/RealGameData/data/observations.jsonl`，读取方法见[数据说明](../Background/RealGameData/README.txt)。统计单位为实际页面、货架或动作实例，不把日志行数当独立抽样数。

从仓库根目录运行：

```powershell
python -m pip install -r Research/analysis/requirements.txt
python Research/analysis/realgame_shop_stats.py
python Research/analysis/realgame_events_analysis.py
python Research/analysis/realgame_notes_profile.py
python Research/analysis/realgame_schema_profile.py
python Research/analysis/realgame_notes_coverage.py
python Research/analysis/realgame_capture_plan.py
python Research/analysis/realgame_write_report.py
```

以上命令会读取总数据并覆盖对应派生文件。`realgame_notes_coverage.py`的5项完整音符／幸运观测结论绑定本次源数据SHA；`realgame_capture_plan.py`、`realgame_write_report.py`保留本次样本计数和审阅摘要。新批次应同时更新人工结论与报告口径；规则目录中的工作概率与实测结论保持分别标注。
