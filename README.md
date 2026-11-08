# Stella Sora Tower Research

Stella Sora 爬塔规则、实机数据分析与潜能价值网络训练。

## 阅读入口

- [潜能状态空间](Background/StateSpace.md)
- [音符状态空间](Background/NoteStateSpace.md)
- [状态空间观察清单](Background/StateSpace_Observation_Targets.md)
- [规则目录与实机分析](Research/README.md)
- [实机总数据说明](Background/RealGameData/README.txt)
- [实机总数据下载](https://github.com/dango636/StellaSora-Tower-Research/releases/tag/real-game-data)
- [训练器使用说明](NeuralNetwork/trainer/README.md)
- [初始模型](models/README.md)

规则文档区分确定规则、实机数据支持的规律和待确定的机制。当前价值网络学习普通潜能收益；音符资料用于完善规则与数据采集。

## 目录

```text
Background/                 潜能、音符规则和规范化实机数据说明
Research/data/              音符规则目录与实机统计结果
Research/reports/           统计分析报告
Research/analysis/          实机数据分析脚本
NeuralNetwork/trainer/      模拟器、价值网络、训练与监控页面
models/                     新训练的统一初始模型
```

## 开始训练

按[训练器说明](NeuralNetwork/trainer/README.md)安装依赖并启动一个新的运行目录。

新运行以 `models/initial.npz` 初始化学生和全部4960个预设的教师，首次启动完成基准评测，随后按预设分别晋升教师。正式选模对每个候选评测4960个等权结构，每个结构256局。完整配置与模型元数据随源码提供。

## 实机数据

`Background/RealGameData/` 维护数据说明、当前计数、预设和潜能字典。

在 [Releases 下载实机总数据](https://github.com/dango636/StellaSora-Tower-Research/releases/tag/real-game-data)，将 `real-game-data.zip` 解压到仓库根目录，即可得到：

```text
Background/RealGameData/data/observations.jsonl
```

数据包包含401040条记录、预设与潜能字典，以及字段和使用口径说明。代码仓库中的[数据说明](Background/RealGameData/README.txt)和当前计数与数据包配套。

## 实机软件

实机插件、已部署模型与安装器维护在 [MaaStellaSora-NeuralNetwork](https://github.com/dango636/MaaStellaSora-NeuralNetwork)。

Maa 的流程、Agent 和区服适配维护在 [MaaStellaSora](https://github.com/MaaStellaSora/MaaStellaSora)。

研究模型的训练更新与实机软件发布分别管理。
