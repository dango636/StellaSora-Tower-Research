# Stella Sora Tower Research

用实机对局研究星塔的潜能与音符规则，并在模拟器中训练、比较爬塔策略。游戏内采集与操作在 [MaaStellaSora](https://github.com/MaaStellaSora/MaaStellaSora) 主项目的原生分支开发；本仓库提供规则、研究工具和评测结果。旧[外置插件](https://github.com/dango636/MaaStellaSora-NeuralNetwork)已停止维护。

## 正在推进的工作

| 方向 | 当前进度 |
| --- | --- |
| 根据预设预测终局潜能 | 小模型已完成两版实验。输入三名角色的 36 项目标等级，输出固定 M215 策略下终局达到各等级的概率，可计算期望与「90+」概率。随机 10 组预设、每组模拟 1 万局，均值平均相差 0.33 级，高分尾部仍偏高。它不替玩家选卡。[评测与逐级概率](Research/reports/PresetForecast.md) |
| 潜能＋音符联合决策 | 联合价值模型已在本地训练，目标是普通有效潜能等级加上「协奏倍率 × 协奏等级」。当前覆盖 4,960 组结构，倍率可取 0–5。本次仅公布研究进展，联合训练器源码和权重暂不发布。 |
| 潜能与音符状态空间 | 两份规则都有可用于模拟的版本。后续拿到更多真实对局，再检验[潜能观察项](Background/StateSpace_Observation_Targets.md)和[音符观察项](Background/NoteStateSpace_Observation_Targets.md)，优先看能区分不同解释的样本，例如库存不均时随机扣除 10 枚音符。 |
| 主项目：未知秘纹 | 正在推进新秘纹尚未进入本地字典时的支持：利用背包可见的协奏等级和下一等级所需音符继续爬塔。不能确认的需求保持未知。 |
| 主项目：从任意界面接续 | 正在推进任务自行识别当前页面，从可读取的稳定界面恢复本局状态后续跑。目前主项目首版候选只开放从主页启动，任意界面接续尚未作为发布功能。 |

## 数据与分析

- [下载实机总数据](https://github.com/dango636/StellaSora-Tower-Research/releases/tag/real-game-data)。新版提供 SQLite 八表数据库、数据说明、快照计数及字典，解压后数据库位于 `Background/RealGameData/database/research.sqlite3`。附件格式及统计以该次 Release 说明和包内 `SUMMARY.json` 为准。
- [数据库结构与使用说明](Background/RealGameData/README.txt) · [导入原生对局](Research/analysis/import_runs.py) · [按动作审核音符观测](Research/analysis/NOTE_QUALITY.md)。总库只收录明确开局及确认接续的段，无开局或未归属片段不入库。同一动作的重读不当作多个样本；普通战斗采用本塔可见的九类，评估随机扣除时单列十三类库存都已知的样本。
- [潜能状态空间](Background/StateSpace.md) · [音符状态空间](Background/NoteStateSpace.md) · [实机分析与静态目录](Research/README.md)。报告会列出自己的样本范围，后续增加的对局不会自动计入以前的研究结论。

## 已公开的训练器

仓库内的[训练器](NeuralNetwork/trainer/README.md)和[初始权重](models/README.md)对应已公开的纯潜能版本。该版本在 4,960 组等权预设上评测，每组 256 局；按训练器说明可以启动新的运行。研究模型的训练与 Maa 实机任务的发布分别管理。
