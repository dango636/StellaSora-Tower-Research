# Stella Sora Tower Research

这个仓库研究《星塔旅人》的星塔玩法。我们用实机对局核对潜能抽取、商店和音符的规则，再用模拟对局比较不同的爬塔策略。

## 规则和实机数据

- **游戏规则**讲解潜能怎么抽取和强化、商店有哪些选择，以及音符怎样激活和升级协奏。 [潜能规则](Background/StateSpace.md) · [音符规则](Background/NoteStateSpace.md)
- **实机数据库**记录开局预设、对局中的观察与操作、结算结果，可用于核对规则。 [数据下载](https://github.com/dango636/StellaSora-Tower-Research/releases/tag/real-game-data) · [数据说明](Background/RealGameData/README.txt) · [研究索引](Research/README.md)

## 纯潜能模型和训练器

开局预设指定了想要的潜能等级。模型为选潜能、强化、重抽等操作打分，目标是通关时尽量满足预设。训练器可以从公开权重接着训练，也可以用相同预设和随机种子比较两套策略的整塔成绩。

[模型权重](https://github.com/dango636/StellaSora-Tower-Research/releases/download/cadenza-ii/Cadenza-II.pt) · [训练器下载与启动说明](https://github.com/dango636/StellaSora-Tower-Research/releases/tag/cadenza-ii) · [整塔评测](Research/reports/cadenza-ii/RESULTS.md)

另有一项研究根据开局预设估计终局的潜能等级。 [预设预测评测](Research/reports/PresetForecast.md)
