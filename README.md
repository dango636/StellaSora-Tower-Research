# Stella Sora Tower Research

《星塔旅人》（*Stella Sora*）星塔玩法的机制研究、实机数据库与爬塔决策模型仓库。

本仓库负责**拆解并验证游戏规则、公开实机研究数据、训练并评测爬塔策略模型**；游戏内自动爬塔与数据采集维护在主项目 [MaaStellaSora](https://github.com/MaaStellaSora/MaaStellaSora)。

## 项目包含什么

| 模块 | 在做什么 | 入口 |
| --- | --- | --- |
| **游戏规则规格** | 将潜能抽取、商店、事件与音符/协奏机制整理成可执行规格，并持续用实机对局核对边界条件 | [潜能规则](Background/StateSpace.md) · [音符规则](Background/NoteStateSpace.md) |
| **实机研究数据库** | 汇总脱敏后的多区服实机爬塔对局（SQLite），用于验证机制假设与做统计分析 | [数据下载](https://github.com/dango636/StellaSora-Tower-Research/releases/tag/real-game-data) · [数据说明](Background/RealGameData/README.txt) |
| **纯潜能模型与训练器** | 根据开局潜能目标预设，为选卡、强化、重抽、商店与事件决策打分；支持续训与同条件策略对比 | [模型权重](https://github.com/dango636/StellaSora-Tower-Research/releases/download/cadenza-ii/Cadenza-II.pt) · [训练器与启动说明](https://github.com/dango636/StellaSora-Tower-Research/releases/tag/cadenza-ii) |
| **分析报告与工具** | 汇整策略评测、音符机制检验、秘纹目录对照、开局终局预测等报告，并提供对局导入与质检脚本 | [研究索引](Research/README.md) |

## 规则规格与机制分析

规则文档采用**可执行规格**与**实测观察报告**分开维护的方式：本体规格只保留唯一一套模拟规则；所有统计检验、待定边界与后续采集目标集中在观察报告中。

- **潜能与商店**：[潜能状态空间](Background/StateSpace.md) · [潜能观察报告](Background/StateSpace_Observation_Targets.md)
- **音符与协奏**：[音符状态空间](Background/NoteStateSpace.md) · [音符观察报告](Background/NoteStateSpace_Observation_Targets.md)
- **专题报告与静态目录**：[研究总索引](Research/README.md) 收录了[音符实机机制分析](Research/reports/NoteRealGameValidation.md)、[秘纹目录线上对照](Research/reports/DiscCatalogCoverage.md)以及 `Research/data/` 下的秘纹、音符与事件静态规则表。

## 实机数据库与离线工具

规范化实机数据库（SQLite）托管在 [real-game-data Release](https://github.com/dango636/StellaSora-Tower-Research/releases/tag/real-game-data)，包含开局预设、对局观测与操作、结算结果及质量标记，已做脱敏处理且不提供教师标签。最新快照计数见 [SUMMARY.json](Background/RealGameData/SUMMARY.json)，表结构与使用口径见 [数据说明](Background/RealGameData/README.txt) 与 [research_schema.sql](Background/RealGameData/research_schema.sql)。

下载解压后数据库位于 `Background/RealGameData/database/research.sqlite3`。仓库附带两个仅依赖 Python 标准库的命令行工具：

```bash
# 只读审核音符观测质量（支持 SQLite 或单局 ZIP）
python Research/analysis/note_quality.py Background/RealGameData/database/research.sqlite3

# 将指定原生对局文件（JSON / GZIP / ZIP / SQLite）去重合并到指定数据库
python Research/analysis/import_runs.py --database combined.sqlite3 path/to/run.json
```

详细用法与统计口径见 [NOTE_QUALITY.md](Research/analysis/NOTE_QUALITY.md) 和 [IMPORT_RUNS.md](Research/analysis/IMPORT_RUNS.md)。

## 纯潜能模型与训练器

开局预设指定了本局想要凑齐的目标潜能等级。价值网络根据当前局面为选潜能、商店购买潜能饮品、强化、重抽与事件选项打分，目标是最大化通关时的有效潜能等级期望。

- **公开权重与训练器**：大文件托管在 GitHub Releases，可直接下载 [公开权重](https://github.com/dango636/StellaSora-Tower-Research/releases/download/cadenza-ii/Cadenza-II.pt) 和 [冻结训练器（含环境与启动说明）](https://github.com/dango636/StellaSora-Tower-Research/releases/tag/cadenza-ii)，用于基于公开权重继续训练，或在相同预设与随机种子下对比不同策略的整塔表现。
- **策略评测**：完整的冻结权重筛选与配对评测结果见 [整塔评测报告](Research/reports/cadenza-ii/RESULTS.md)。
- **附加研究**：另有一个轻量预测模型，用于根据开局预设估计终局潜能等级分布，详见 [预设预测评测](Research/reports/PresetForecast.md)。

## 仓库结构

```text
StellaSora-Tower-Research/
├── Background/
│   ├── StateSpace.md                           # 潜能与商店状态空间规格
│   ├── StateSpace_Observation_Targets.md       # 潜能机制证据与待定观察项
│   ├── NoteStateSpace.md                       # 音符与协奏状态空间规格
│   ├── NoteStateSpace_Observation_Targets.md   # 音符机制证据与待定观察项
│   └── RealGameData/                           # 实机库 schema、数据说明、元数据与最新计数
├── Research/
│   ├── README.md                               # 研究资料与报告总索引
│   ├── data/                                   # 静态规则表、秘纹/音符目录与聚合结果
│   ├── reports/                                # 策略评测、音符分析、预设预测与目录对照报告
│   └── analysis/                               # 原生对局合并（import_runs.py）与音符质检（note_quality.py）
└── GitHub Releases（大文件不纳入 Git）
    ├── 模型权重与冻结训练器
    └── 实机研究数据库（SQLite）
```

## 与实机自动化的关系

- **游戏内运行**：想直接在游戏中使用神经网络自动爬塔，请前往主项目 [MaaStellaSora](https://github.com/MaaStellaSora/MaaStellaSora)；研究模型的训练与主项目发布分开管理。
- **历史仓库**：早期外置插件 [MaaStellaSora-NeuralNetwork](https://github.com/dango636/MaaStellaSora-NeuralNetwork) 已停止维护。
- **当前公开范围**：当前公开的权重与训练器为**纯潜能策略**；音符与协奏部分目前提供规则规格、实机数据与统计分析，潜能＋音符联合模型的训练器与权重暂不发布。

## 致谢

- 感谢 [MaaStellaSora](https://github.com/MaaStellaSora/MaaStellaSora) 项目提供的自动化框架与实机采集基础。
- 感谢北辰（[beichen24a1](https://github.com/beichen24a1)）在 [PR #1](https://github.com/dango636/StellaSora-Tower-Research/pull/1) 中提出的未知不补零、按动作去重与冲突整组排除等质检思路（详见 [NOTE_QUALITY.md](Research/analysis/NOTE_QUALITY.md)）。
