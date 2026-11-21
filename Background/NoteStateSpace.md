# 星塔爬塔：音符状态空间

## 0. 模型设置

本文规定音符的完整输入、真状态与可见状态、动作、包级奖励、阶段配额、库存、协奏、事件、货架、计分和快照续跑。与潜能联合运行时，[潜能状态空间](StateSpace.md)继续定义路线、潜能、金币、强化及事件类型生成；本文扩展音符库存与相关结果，不另维护潜能规则副本。

模型版本：`note_state_v5_20261115_r3`。本文给出一套完整的生成与更新程序：共享非首领额外包预算、首领二态记忆、包级惊喜，以及按来源分开的持久类别计数。证据分辨率和后续观察项目集中在[音符状态空间观察报告](NoteStateSpace_Observation_Targets.md)，不参与本体的运行分支。

### 0.1 固定条件

| 项目 | 本规格采用的设置 |
|---|---|
| 研究 | 星塔研究全部点满；不区分逐账号解锁，不增加研究节点OCR |
| 路线 | 潜能规格的20层、14战、4店路线；最终商店结束才结算 |
| 队伍 | 一名主控、两名援护，记录元素、三个主秘纹及全部辅助秘纹 |
| 音符支付 | 共享任务默认5／10枚音符选项始终可支付；具体扣取按§6.2执行 |
| 战斗 | 模拟奖励与决策过程，通关不由HP、伤害或音符效果另行判定 |
| 核心与潜能 | 原始预设、目标0项、满级6、核心固定加6等沿用潜能规格 |
| 三塔对偶 | 光土／水风／火暗共用音符过程，只替换元素及输入目录 |
| 随机机制 | 每局初始化隐藏状态，按动作顺序推进并生成新轨迹 |
| 实机缺失 | 未显示／未读到的类型为null，明确读到0才为0；模拟器完整真库存为整数向量 |

难度≥8且主控元素属于本塔双元素时，D保留七种通用音符与主控元素；否则沿用原始掉落分组。该目录条件已有研究40101的设计文本及字段依据。元素类型与任一单一通用类型等权（1∶1）已获实测支持，类别生成按§5.1执行，模型与配置随样本保存。[E27]

### 0.2 游戏状态、玩家状态和机制配置

游戏真状态含库存、共享总预算、阶段位置、惊喜失败数、首领恢复标志、来源／子池抽类计数及RNG。玩家状态含当前公开库存、菜单／货架、历史和隐藏状态的相容路径或后验；策略不能读取未公开的配额位置或未来随机结果。预算概率、位置权重和类别先验等机制配置随模型版本固定，不与某一局隐藏状态混为一体。

生成器可以保存内部诊断真值；策略输入使用公开观察，后续需要续局分析时从完整快照克隆，不把隐藏真值混进策略输入。

## 1. 完整输入与目录

### 1.1 静态目录

静态目录是运行输入。下列外部路径规定项目中的目录与接口映射；本规格不表示对应外部仓库已经部署本版本。

[NoteRules.json](../Research/data/NoteRules.json)包含13种音符、771条协奏、106件秘纹、118组／1062条阶段供给。214068「帐幕下的酣梦」与214069「紧急救援」由国服资源2611122000、日服资源2611122238的相同配置增补，均含协奏1–5级及阶段0–8供给；旧104条及2026-11-08线上对照保留原来源。效果及定点量化见[效果目录](../Research/data/stella_note_effects_verified.json)，协议见[NoteProtocol.json](../Research/data/NoteProtocol.json)。目录保留源文件、记录偏移和字段语义，供给按具体秘纹与阶段绑定，不由旧PR未绑定阶段的数值覆盖。

当前国服2.1.0／资源2611131417已恢复13类音符、完整45条Character.EET、三塔掉落分组和研究40101，静态目录缺口关闭。90011–90017依次为强攻、幸运、暴发、体力、专注、技巧、绝招之音；90018–90023依次为水、火、风、地、光、暗之音。角色元素按Character.EET查询：1水、2火、3地、4风、5光、6暗；角色162、115、107均为5（光）。任意队伍仍以实际主控角色字段为输入，不从音符未出现反推元素。[E27]

本塔原始掉落集合由SubNoteSkillDropGroupId提供：[E1、E9、E27]

| 组 | 原始音符集合 |
|---|---|
| 1 | 90011–90017 |
| 2 | 90011–90017、90021、90022 |
| 3 | 90011–90017、90018、90020 |
| 4 | 90011–90017、90019、90023 |

难度8的StarTower记录109／209／309分别为光土／水风／火暗，EET分别为[5,3]／[1,4]／[2,6]，对应掉落组2／3／4。研究40101的字段为IsServer=1、EffectClient=32、Clientlvl=8，设计文本规定主控与塔属性一致时只出现同属性元素音符；本体据此构造有效D，七通用保留。完整目录和角色映射已确认；元素与单一通用类型的权重比为1∶1，实测支持已收拢至观察报告S17，不再列为待定。[E27]

库存始终保留90011–90023全部13种。辅助供给的塔外元素照常入库；掉落池约束新增随机音符，不裁掉已有库存，也不约束兑换扣取池。

### 1.2 单局输入

| 输入 | 内容 |
|---|---|
| 模型与资源 | 模型版本、配置、客户端／资源版本、规则目录版本、随机种子 |
| 塔与路线 | StarTowerId、难度、掉落分组、主控元素、研究条件、共享路线 |
| 队伍与秘纹 | 角色位置与元素；主／辅秘纹ID、Level、Phase、Star；主协奏组从目录查询 |
| 潜能任务 | 完整合法目录、有序六池、每项原始目标向量及共享潜能模型输入 |
| 初始状态 | 初始币、材料、可选完整13维库存；是否已含辅助供给和开局奖励 |
| 分项指标 | 潜能有效等级、音符构筑分、协奏构筑分分别保存；带音符价值模型的训练目标见 §2.2 |

进入当前价值训练范围的数据，原始潜能目标向量仍逐样本遵守4960个等权结构约束。音符输入不扩大该范围，不把剩余目标或当前得分当作原始预设。独立音符研究样本记录自己的任务配置。

### 1.3 辅助供给

每个辅助秘纹按SubNoteSkillGroupId与Phase，查找当前阶段及以下最近一条供给记录，直接取整条记录，各辅助供给相加：[E12]

\[
q_g(p)=Supply(g,\max\{j:j\le p\}),\qquad
\mathbf n_{aux}=\sum_{d\in auxiliary}q_{g_d}(Phase_d).
\]

供给不是逐阶段累加。已有进塔／续局快照包含供给时直接载入；从零生成时在开局奖励前只发放一次。

## 2. 计分、奖励与目标

### 2.1 音符分与协奏分

音符构筑分：[E9]

\[
V_{note}(s)=15\sum_{i=90011}^{90023}\min(n_i,99).
\]

对每个主秘纹关联的协奏组g，从所有音符需求均满足的行中选最高等级a_g；没有满足行则未激活：

\[
a_g(\mathbf n)=\max\bigl(\{0\}\cup\{l:\forall i,\ n_i\ge Need(g,l,i)\}\bigr),
\qquad
V_{concerto}(s)=\sum_{d\in main}\sum_{g\in groups(d)}Score(g,a_g).
\]

等级0记未激活，Score(g,0)=0。求和按主秘纹持有项及其组处理；不能把三张主秘纹需求的并集当成一条协奏。150个正常组各有5级，另有特殊记录9001；特殊行按目录与秘纹绑定使用，不自动分给全部队伍。316行需要两种音符、435行需要三种。常见门槛10／25／40／55／70或15／30／45／60／75，实际值逐行查表。

A_local由库存重算；A_server是实机服务器下发的激活ID集合，分开保存。实机终局采用返回的A_server；模拟局按本规则计算并标明来源。[E1、E2、E12]

### 2.2 任务奖励

潜能总有效等级继续为Z=6+Σmin(t_i,l_i)，不把游戏构筑分换算为潜能等级。生成器沿用当前价值模型的非核心口径，输出potential_effective=Z−6、note_build=V_note、concerto_build=V_concerto三个分项；每步同时输出各分项delta，终局输出完整分项结果。核心固定6另加即可，不进入原始普通目标或结构范围。

带音符价值模型最大化终局收益 `R = potential_effective + 2 × concerto_level_sum`。在最终商店结束后，按每件主秘纹绑定的每个协奏组取终局已达到的最高等级，未激活记 0，再相加得到 `concerto_level_sum`。裸音符库存和游戏显示的协奏构筑分不直接计入 R；中途激活、升级也不重复计奖。

独立生成器默认随机选择合法动作，保留原始分项、终局音符库存及秘纹配置，可据此重算各组等级。兑换和购买均按真实支付执行，允许操作后某个分项得分下降。

## 3. 完整状态与观察更新

### 3.1 完整真状态

\[
S=(I,P,node,f,phase,\mathbf n,c,m,h,A_{server},O,H,E,B,K,\rho).
\]

| 分量 | 内容 |
|---|---|
| I | 塔、队伍、秘纹、目录、原始目标、模型配置 |
| P | 共享潜能完整状态：持有、六池、页面、来源、展示计数、强化／金币／NPC计数及公开信念 |
| node/f/phase | 下一条路线指令、最近楼层、当前交互阶段；楼层不能替代指令位置 |
| n | 13种实际库存；真值与可读完整性分开 |
| c/m/h | 金币、材料、生命及最大生命／相关效果；HP不另判战斗胜负 |
| A_server | 实机下发的激活协奏；模拟计算结果另有来源标记 |
| O | 当前公开观察、上一可见库存、来源／完整性、待处理变化、动作顺序 |
| H | 店序、初始／补货、货架身份、后台完整8件商品、已购集合、刷新许可／费用／全局剩余次数（初始2） |
| E | 事件类型、NPC、题目、选项ID与显示顺序、禁用项、完成状态、本局已见集合 |
| B | 构筑ID、锁定／限时试用状态、修改状态、修改前库存及材料 |
| K | 下表中的音符隐藏状态 |
| ρ | 全部随机流当前状态，保留后续抽取位置 |

实机每件商品保留Sid、Idx、Type、GoodsId、CharPos、Tag、音符类型及数量、原价和实际价；生成器的对应字段见§9。每次操作保留对局ID、动作ID、操作顺序和读取来源。

### 3.2 音符隐藏状态

| 字段 | 初始与更新 |
|---|---|
| supply_issued | 辅助供给是否已包含，续局不重复发放 |
| opening_issued | 开局奖励是否处理，独立于首战奖励 |
| surprise_used | 已触发惊喜次数0–4；跨房型／阶段保存 |
| surprise_failures | 封顶前连续失败包数0–4；失败加1、成功归零；封顶后与surprise_used一起规范为(4,0)，见§5.5 |
| boss_need_recovery | 初始false；首领失败置true、成功置false；仅保留下一次概率所需的二态记忆 |
| bonus_budget | 每局一次抽取T∈{2,3,4}，作为全部非首领额外包总预算；不是三个阶段各自独立抽数 |
| early_state | T=4时有1个名额，保存未处理位置或等价的剩余条件分布 |
| middle_state | T≥3时有1个名额；与early_state共享同一T，保留未处理位置或剩余条件分布 |
| late_state | 末段恰2个名额的剩余数量及未处理位置／抽签状态 |
| type_draw_counts | 按来源 B/E、子池 C/N 保存各类型抽取计数；初始为0，按§5.1更新；单池退化时使用对应来源的全D计数 |
| reward_context | 当前来源、包数、包内抽类与惊喜进度，支持奖励中断快照 |

本表规定状态语义。初始化先设抽类计数与惊喜计数，再抽共享T和各阶段位置，发放供给，最后处理开局包，具体顺序见§9。潜能重抽、背包读取及续跑不重置这些计数。

### 3.3 可见状态与后验

observe仅提供当前已公开的库存及完整性、协奏、金币、潜能页、事件菜单、货架和合法动作。未公开的配额位置、包顺序、失败真值、下一首领结果和RNG不进入策略输入。

聚合变化使用全部相容分解：同类基础包合并显示，分解由整场包数、升级次数与完整类型向量共同约束。隐藏包顺序、惊喜计数、阶段位置和抽类历史计数一起进入后验，不把弹窗条目数当作抽取次数。设b_t(k)为公开隐藏状态后验：

\[
b_{t+1}(k')\propto\sum_k b_t(k)P(k',o\mid k,a_t,I).
\]

缺少包级先后时对隐藏顺序求和；相同公开结果的后台路径合并。实机缺失只限制依赖该字段的推导，不遮蔽其他独立清晰字段，观测记录与内部生成真值分开保存。正常路线第18层战斗结算完成后，所有相容路径的惊喜状态均合并为(4,0)，无需逐包实读；路线前缀验证条件见§5.5。

### 3.4 快照与变化量

进塔／续局Bag.Items是赋值，交互响应Data.Infos逐条累加：[E1]

\[
n_{Tid}\leftarrow Qty\quad（快照）, \qquad
n_{Tid}\leftarrow n_{Tid}+Qty\quad（增量）.
\]

库存按数量直接累加，不截断到99，作为硬规则执行：库存10收到+5后为15，收到−5后为5；库存98收到+5后为103，不截为99，也不拒收奖励。实机按响应Qty更新，模拟按生成增量更新。快照、战斗、事件、购买分别处理。以供给页为基线的首层差分联合包含开局包与首战包；按两类包的完整生成过程更新观察后验。[E29]

### 3.5 面向目标的可解释表示：需求、缺口和溢余

完整真库存保留13维。对每个主秘纹协奏组g、等级l、需求种类i，计算：

\[
d_{gli}=\max(Need(g,l,i)-n_i,0),\quad
L_g=\max\bigl(\{0\}\cup\{l:\sum_i d_{gli}=0\}\bigr).
\]

公开派生特征包括当前等级、下一级逐类缺口、缺口种类数、达到各级所需的最小额外枚数及逐类溢余。相同需求的不同组共享计算结果，仍按各自绑定次数计分。

部分库存以区间表示：已知n给[n,n]，未读给[0,+∞)。把上下界分别代入静态需求，得到等级区间[L_g^−,L_g^+]。区间退化为一点时输出确定等级；例如某级需要三类各15枚，其中一类已知只有11枚，即使另有类型未读，该级仍不可达。区间下界不写回库存真值。

对需求种类相同、各类首级需求均为b_g、以后每级增加15的组，采用下式计算等级；其他组逐行查询Need：

\[
L_g=\operatorname{clip}_{0,5}\left(1+\left\lfloor\frac{\min_{i\in J_g}n_i-b_g}{15}\right\rfloor\right).
\]

总音符数、当前等级总和不能替代逐类缺口。评价某次奖励时，把奖励向量加入库存后重算各组等级；结算后的数学比较不新增可执行动作。

按枚扣取使溢余库存参与损失缓冲。库存不在协奏门槛处截断，也不把非需求类别合成一个桶；兑换、商品身份、库存上界及部分观察统一使用完整13维。共享T、阶段位置、首领恢复标志、惊喜状态、来源类别计数及其公开后验共同决定未来。

## 4. 路线、动作与顺序

共享路线及核心／普通奖励顺序使用[潜能规格§2](StateSpace.md#2-完整楼层与执行顺序)。音符挂到同一BATTLE指令，在战斗结算时发放，随后处理核心、普通奖励和追加NPC。非首领为1／3／4／7／8／10／11／14／15／17／18层，首领为6／13／20层。

| 动作／自动转移 | 音符侧结果 |
|---|---|
| BATTLE | 自动执行§5，更新库存及隐藏计数，再回共享路线 |
| SELECT／REROLL | 使用共享潜能转移，不改音符计数／配额 |
| SELECT_EVENT(index) | 按当前零基显示索引找具体选项；支付／获得及潜能创建属于同一次结果 |
| CLOSE_EVENT | 免费结束；非121事件保留出现时已加入的seen，121不加入seen |
| BUY(Sid) | 扣实际价格、标记已购；音符立即入库，饮料进入奖励页，处理后返回原货架 |
| ENHANCE | 共享强化转移，不改音符过程 |
| RESTOCK | 扣100、次数减1，整架按E6替换；库存及战斗计数保留 |
| LEAVE | 离店，最终店离开后END |
| MODIFY_NOTES(Δ) | 满足构筑入口时执行§8.2；独立于正常爬塔终局 |

动作使用当前页面、金币和许可；未处理的潜能页不能跨过去继续买下一件。随机音符类型不是玩家动作，兑换不允许自行选择损失最小的扣取分配。

## 5. 奖励生成程序

### 5.1 类别池与来源计数

D为掉落池经40101元素条件处理后的有效集合，由塔、难度、主控元素和固定满研究条件确定。C为三个主秘纹全部协奏组所有等级需求ID的去重并集与D的交集，N=D\C。C在本局固定；已激活、当前等级已满足或满级协奏均不删除其需求类型。

同一候选子池内，元素类型与任一单一通用类型的基础权重比为1∶1，已获实测支持，不另设元素倍率。主需求池4/5选择与后续历史计数更新仍按下述程序执行。

类别流分为两个来源：

| 来源 | 使用范围 | 计数单位 |
|---|---|---|
| B | 开局包和全部战斗包，包括首领附加包 | 每个基础包抽类一次、对应类型计数加1 |
| E | 实际获得的随机音符事件奖励 | 每枚抽类一次、对应类型计数加1 |

对每个来源s及非空子池P，保存整数计数c_i^(s,P)，初始均为0。若C和N均非空，先以4/5选择C、1/5选择N；再从所选来源、子池中按下式抽取类型i，并把对应计数加1：

\[
P(i\mid s,P,\mathbf c)=\frac{7+c_i^{(s,P)}}{7|P|+\sum_{j\in P}c_j^{(s,P)}}.
\]

C为空或C=D时，直接使用该来源的全D计数，不再抽主池分支。两个来源及各子池分别维护计数，局内一直保留。这里7是每类参数；K类池的总浓度为7K。该计数核给出的有序序列分布为：

\[
P(x_{1:n})=\frac{\Gamma(7K)}{\Gamma(7K+n)}\prod_{i\in P}\frac{\Gamma(7+c_i)}{\Gamma(7)}.
\]

这定义了每来源、每子池独立Dirichlet(7,…,7)的积分生成程序；执行时直接保存计数。3枚包与9枚升级包都只发生一次类别抽取。指定奖励、菜单生成、辅助供给、商店购买与兑换扣取均不更新B/E计数；兑换也不倒减已发生的历史抽类计数。

类型分解未被公开观察完全确定时，玩家状态按全部相容计数分支更新belief。模拟器记录实际生成的抽类计数；公开库存不能直接除以3来恢复这些计数。

### 5.2 开局与跨阶段共享总预算

初始化惊喜状态为 `surprise_used=0`、`surprise_failures=0`。确定性辅助供给只发一次，不推进惊喜与类别计数。随后发放一个独立的3枚开局包，按来源B抽类并执行§5.5惊喜；处理完成后设置 `opening_issued=true`，再执行第1层战斗。

每局只抽一次全部非首领额外包总预算：

\[
T\sim\operatorname{Uniform}\{2,3,4\},\qquad
Q_{early}=\mathbf1(T=4),\quad Q_{mid}=\mathbf1(T\ge3),\quad Q_{late}=2.
\]

| T | 早期1/3/4层名额 | 中段7/8/10/11层名额 | 末段14/15/17/18层名额 | 概率 |
|---:|---:|---:|---:|---:|
| 2 | 0 | 0 | 2 | 1/3 |
| 3 | 0 | 1 | 2 | 1/3 |
| 4 | 1 | 1 | 2 | 1/3 |

早期每场有1个基础包。Q_early=1时，按权重[2,2,1]在第1/3/4层选择一个位置，该场再加1包。条件位置概率为40%、40%、20%；各层额外包的无条件边际概率为2/15、2/15、1/15。

### 5.3 中段、末段位置与公开后验

Q_mid=1时，从第7/8/10/11层按[2,3,3,2]选择一个位置，条件位置概率为20%、30%、30%、20%。中段每场有1个基础包，选中位置再加1包。中段名额由T决定，边际概率为2/3。

末段在第14/15/17/18层按[4,4,3,3]逐次加权无放回选两个位置：第一次按原权重抽，移除该位置，再按剩余权重抽第二次。每场有1个基础包，命中位置各加1包。六个无序位置对的概率为：

| 位置对 | 概率 |
|---|---:|
| (14,15) | 8/35 |
| (14,17)、(14,18)、(15,17)、(15,18) | 各9/55 |
| (17,18) | 9/77 |

策略使用公开历史后验。已经明确观察到早期额外包时，T=4，中段有名额的概率为1。在中段开始前，早期三场及来源均已明确且没有额外时，T∈{2,3}且各半，中段有名额的概率为1/2。早期未结束、缺读或来源聚合时，对全部相容T、阶段位置、类别计数和惊喜状态联合求和。

对相容位置集合A及已观察前缀h，下一场j命中的条件概率为：

\[
P(j\in A\mid h)=\frac{\sum_{A\sim h,j\in A}P(A)}{\sum_{A\sim h}P(A)}.
\]

内部状态保存预抽位置，公开belief保存相容位置的条件分布。末段后验同时包含已观察位置与未处理位置对，不仅保存剩余名额数。

### 5.4 首领幸运：恢复标志二态模型

首领出现在第6/13/20层，每场有2个基础包；幸运成功再增加3个基础包。恢复标志初始为false：

\[
P(B_j=1\mid recovery)=
\begin{cases}
1,&recovery=true,\\
3/5,&recovery=false.
\end{cases}
\]

本次失败置 `boss_need_recovery=true`；成功置false。首次首领与前次成功后的首领使用相同的3/5概率。附加3包分别按来源B抽类，全部包都进入惊喜过程。首领幸运结果与协议LuckyLevel按各自字段保存。

### 5.5 包级惊喜与总量

每个开局或战斗基础包先按来源B抽类得到t_j，再执行：

```text
if surprise_used == 4:
    z = 0
else:
    p = min(1, 0.2 + 0.2 * surprise_failures)
    z = Bernoulli(p)
    if z:
        surprise_used += 1
        surprise_failures = 0
    else:
        surprise_failures += 1
notes[t_j] += 3 + 6*z
```

失败数0/1/2/3/4对应20%/40%/60%/80%/100%。成功清零；失败记忆跨战斗、阶段、房型及商店保留。事件、购买和确定性供给不消耗惊喜机会，不增加失败数，也不清零已有计数。第四次惊喜后进入吸收状态(4,0)：后续包保持3枚，不再推进有效失败数。

整场库存增量为：

\[
\Delta\mathbf n=\sum_{j=1}^{b}(3+6z_j)e_{t_j},\qquad
r'=r+\sum_jz_j\le4.
\]

同类包在库存中合并，内部保留真实抽类和机会顺序。首领5包且剩余4次惊喜时，本场奖励上界为39枚。

#### 第18层后的惊喜状态规范化

从任意未封顶状态出发，接下来的5个合格基础包内至少触发1次惊喜：连续失败到4次时，下一包的概率为100%。由新局(0,0)出发，处理完20个合格基础包后必到达(4,0)。

正常路线完成第18层战斗结算时，开局包与此前战斗包数至少为：

\[
1_{\text{开局}}+11_{\text{非首领基础}}+4_{\text{前两首领基础}}
+2_{\text{非首领额外预算下限}}+3_{\text{前两首领至少一次幸运}}
=21.
\]

前两次首领至少一次幸运，因为前次失败使下一次必成功。全部非首领位置已在第18层结算完毕，因此非首领额外包为完整的T。生成器在该结算完成点将惊喜状态统一保存为(4,0)；第19层及之后的事件、购买与最后首领均使用吸收状态。第17层之后仍有只处理19包的合法路径，规范化时点固定在第18层结算完成后。

公开历史导入在确认使用本版本规则、开局只发一次且共享正常路线的全部战斗已按序完成至第18层时，同样把惊喜后验合并为(4,0)。这里依赖已完成的路线前缀；库存实读缺失不妨碍合并。当前楼层数字、正在结算的第18层奖励或自定义续局本身不满足该前提，按实际恢复的动作历史与reward_context继续维护状态。

#### 完整路线的音符总量

正常完整14战、开局只发一次时，四次惊喜已在第18层结束前用尽。设B为三次首领的幸运成功次数，B∈{1,2,3}，则开局与战斗的净获得总量恒为：

\[
N_{open+battle}=3(1+11+3\times2)+6\times4+3T+9B
=78+3T+9B,\qquad T\in\{2,3,4\}.
\]

T独立均匀抽取，首领幸运核只读取前次首领结果，因此在新局生成分布中T与B独立。正常完整路线的开局与战斗来源总量分布为：

| 首领幸运成功次数B | P(B) | N_open+battle的取值 | 所列每个总量值的概率 |
|---|---:|---|---:|
| 1 | 20/125 | 93、96、99 | 每值4/75 |
| 2 | 78/125 | 102、105、108 | 每值26/125 |
| 3 | 27/125 | 111、114、117 | 每值9/125 |

该来源总量的期望为 **105.504枚**，即78＋3×3＋9×(257/125)。辅助供给、事件、购买及扣取分别计入库存，不包含在该分布与期望内；105.504不是终局实际背包库存的期望。中途观测后的T与首领状态按§3.3共同条件化，使用联合后验；路线中途的前缀总量仍按实际已处理基础包数及已触发惊喜次数计算。

## 6. 事件、兑换与货架

### 6.1 事件选项与菜单

| 选项 | 结果 |
|---|---|
| 10301 | 随机消耗5枚，获得150币 |
| 10401 | 随机消耗10枚，创建随机稀有潜能奖励 |
| 10701–10707 | 分别获得90011–90017指定种类5枚 |
| 10708 | 逐枚随机获得5枚 |
| 10801–10807 | 花140币，指定种类10枚 |
| 10808 | 花90币，逐枚随机10枚 |
| 11403 | 答对，逐枚随机10枚 |
| 11401／02／04／05 | 答错，无音符奖励，更新禁用项 |
| 12001 | 花50币，逐枚随机5枚 |
| 12002 | 各1/2恢复30%生命或逐枚随机获得5枚，作为确定规则执行 |
| 12702 | 逐枚随机5枚 |

事件类型、固定／追加NPC及金币／潜能结果沿用共享规格。非121事件出现即加入全局seen，直接关闭也不移除；121不加入seen，每次普通事件抽取都保留其权重。[E3、E13]

107／108从七种通用音符均匀无放回选两种，共21对；指定项按ID升序显示，随机项第三，108的30币项第四。菜单生成不更新类别计数。选择指定种类项时按指定种类入库；选择随机项后，按来源E逐枚抽取。

其他菜单按静态配置生成。114按题目／答案映射判定，答对后按来源E逐枚发10枚；答错更新禁用项。12002固定以各1/2选择恢复30%生命或逐枚随机获得5枚，音符分支使用来源E，作为确定规则执行，不再列待验或继续采样校准。配置证据保留：EventOptions[12002]文案明确50%／50%，EventResult[1200201]含RecoveryHP=14、Parameter1=[989992]，EventResult[1200202]为GetRandomSubNodeSkill=17、Parameter1=[5,1]。实机提交服务器Options的零基显示索引，静态optionId不直接作为显示索引。[E28]

同次随机音符奖励允许重复抽中音符类型，按枚依次抽类并更新来源E的计数；不套用3枚战斗包，也不推进惊喜状态。

### 6.2 随机扣取

按枚无放回：在全部13维实际库存中，每次以n_i/Σn抽一枚、将对应n_i减1，再按剩余库存重算，重复5或10次。塔外辅助供给也参与。总库存N、扣取k时损失向量d服从：

\[
P(\mathbf d\mid\mathbf n,k)=
\frac{\prod_i{n_i\choose d_i}}{{N\choose k}},
\quad\sum_id_i=k,\quad0\le d_i\le n_i.
\]

默认配置以辅助秘纹214009／214024的Phase=8供给28枚，再处理开局奖励，满足默认数据的初始支付条件。自定义配置仍按真实库存执行：库存不足5／10时相应动作不可执行，不补造音符、不记负库存。共享潜能抽象中的“始终可支付”在音符过程通过足额配置与真实合法动作实现。

未显示的库存维度保留缺失掩码；观测更新对所有相容实际库存求和，不把塔外未读库存填0。

### 6.3 共享E6商店

每架8件；数量和折扣沿用[潜能规格的E6程序](StateSpace.md)。小包5枚原价90，大包15枚原价400，饮料原价200。音符商品的身份抽取使用有效D均匀iid，不使用4/5主需求门，也不更新B/E类别计数。

E6先生成额外主饮Y、大包标志B和额外通用饮料X。Y/B的联合概率按(00,01,10,11)为(1,5,2,7)/15。令b=int(shopOrdinal==1)：

\[
G=2-b+X,\quad M=b+Y,\quad X=\min(2,\operatorname{Binomial}(4,.65-.05Y)),
\]

音符商品件数为 `N=6−X−Y`，其中大包件数为B，小包件数为N−B。首店补货仍使用首店数量规则。

**先生成N件音符商品的身份，再分配大包标签。** 每件身份在D内逐类型等权抽一次，元素类型与任一单一通用类型均为1份权重，允许重复；该等权关系已获实测支持。B=0时全部为5枚；B=1时从这N件音符商品中选一件设为15枚，其余为5枚：

- 本局第一次实际RESTOCK：burst（90013）商品的标签权重为3/2，其余商品为1。
- 其他货架，包括全部新店初始架及后续RESTOCK：各音符商品的标签权重均为1。

首次实际补货的判断使用**此次货架生成原因 `cause="RESTOCK"` 且刷新前全局剩余次数为2**。`global_restocks_remaining` 是共享潜能状态 `P.restocks` 的别名。联合环境的RESTOCK只执行一次共享扣款和计数减1；音符扩展读取本次动作前的计数来生成大包标签，不再次扣款或减次数。全局补货预算初始为2，跨店保存，不按商店重置：

```text
RESTOCK action:
    before = global_restocks_remaining
    pay(100)
    global_restocks_remaining = before - 1
    generate_shelf(cause="RESTOCK", restocks_before=before)

Inside generate_shelf(cause, restocks_before):
    first_restock = (cause == "RESTOCK" and restocks_before == 2)
```

后续新店初始架即使剩余次数为1，也不触发首次补货规则。大包标签只在N件音符商品之间选择，按商品件数加权。若其中m件为burst，首次补货的大包为burst的条件概率为：

\[
P(\text{burst大包}\mid N,m,B=1)=\frac{(3/2)m}{N+(1/2)m}.
\]

全部8件商品生成完成后，均匀选4件折扣，再在折扣中选 `D_half=min(2,Binomial(3,p_D))` 件半价，其余八折。第2／3店初始架p_D=3/4，其余初始架和全部补货p_D=7/10。

实际价为 `Discount>0 ? Discount : Price`，按Sid购买并保留商品身份与已购集合。补货重建整架，不重发辅助供给，不重置战斗、惊喜、预算或类别计数。[E3、E5]

## 7. 效果与客户端表现

### 7.1 效果参数

主技能按MainSkillGroupId与Star+1选行，协奏按主秘纹绑定及库存选最高满足行。音符以实际数量为等级参数，主技能／协奏挂到全队。[E2、E12]

| 音符 | 每级参数 | 属性处理 |
|---|---:|---|
| 90011攻击、90014生命 | 0.003×L | PERCENTAGE |
| 90012暴击率 | 0.001×L | BASE_VALUE |
| 90013暴伤、90016技能伤害 | 0.0046×L | BASE_VALUE |
| 90015普攻伤害 | 0.012×L | BASE_VALUE |
| 90017必杀伤害 | 0.01×L | BASE_VALUE |
| 90018–90023六元素伤害 | 0.003×L | BASE_VALUE |

属性合成为(origin+baseAmend)×(1+percentAmend)+absAmend。FDP为有符号64位定点，值raw/2^24；decimal转最近整数，中点远离零。完整13×99参数及量化保留在效果目录。[E17、E20]

当前Config.DiscSubNoteSkillMaxLevel=99。商店OnBtn_Confirm在该类当前库存≥99时禁购音符，判断只看当前数量，不检查“当前数量＋商品数量”；修改界面沿用99限制。音符评分仍按SubNoteSkill.Scores的99项封顶。奖励库存直接累加，允许超过99，不截断、不拒收，作为硬规则执行，不再列待定或安排边界采集。ProcessTowerChangeData按Qty更新，ResetNoteInfo按原数量传给C#；评分、购买、修改限制与实际库存分别处理。[E6、E29]

### 7.2 预览与变化展示

掉落动画使用服务器SubNoteSkillNum作为表现额度，结算补播剩余；库存以交互响应更新。地图表现随机流与奖励随机流分别保存。[E18、E19、E22]

变化弹窗来自当前库存与上次UI缓存差，一战可多次展示。橙色“额外等级”标签读取LuckyLevel>0，绿箭头表示该种变化，二者不提供触发次数。LuckyLevel只从对应协议或明确UI字段读取；生成器的惊喜／首领真值不填入该观察字段。[E17、E26]

### 7.3 满研究节点文本

| 节点 | 文本配置 |
|---|---|
| 10101／10401／20101／20401 | 战斗间音符概率提升15%／30%／45%／60%，后三项要求难度2／3／4及以上 |
| 10102 | 开局额外3枚 |
| 20201／20501／30501惊喜 | 难度3起10%额外+3、最多2次；难度4起10%额外+6、最多2次；难度6起20%额外+6、最多4次 |
| 10701／20701／30701幸运 | 难度3／5／7及以上，首领与魔王后50%概率额外+3／+6／+9 |
| 40101圣洁的共鸣 | 难度8及以上且主控与塔属性一致，只出现主控同属性元素音符，仍保留七种通用音符 |

本表记录固定满研究条件的客户端文本配置。奖励生成与隐藏状态更新统一执行§5。[E26]

## 8. 终止、修改与快照续跑

### 8.1 终止与复现

第20层首领后仍有普通潜能与最终店；最终店离开后END。保存全13维终局库存、激活协奏、各分项得分、原始预设、模型配置与轨迹种子。

快照包含§3完整状态、已打开的页／菜单／货架、公开历史或足够信念，以及全部RNG当前位置。相同快照和相同动作复现相同后续；只存初始seed不足以中途恢复。续局不重发供给、不重抽已公开货架、不重置配额或失败数。

同根多动作比较从各自克隆开始，使用约定的独立续局seed或公共随机流并保存协议；不能把前一动作推进后的状态直接作为下一动作起点。

### 8.2 构筑修改

修改独立于正常爬塔终止。入口：非限时试用、未锁定、构筑分≥5800、修改状态0或1，消耗35013×1；状态2已改音符、3不可修改。[E6、E7、E9]

提交新减旧的Δ，可调换1–15枚：

\[
\sum_i\max(\Delta_i,0)=\sum_i\max(-\Delta_i,0)=k,
\quad0<k\le15,\quad0\le n_i+\Delta_i\le99.
\]

成功采用返回构筑与激活ID。包含修改的任务在输入声明奖励和终止位置；默认爬塔终局不自动增加一次未声明的改音符优化。

## 9. 实现接口与数据映射

本节规定外部项目需要实现的接口。列出的路径是接口映射，不表示这些仓库文件已经部署 `note_state_v5_20261115_r3`。

| 目标路径 | 接口责任 |
|---|---|
| `profiles/note_generation.json` | 保存本版本参数、默认setup、类别来源、先验参数和货架生成原因 |
| `stella/note_rules.py` | NoteRules／NoteSetup；加载13类、静态需求、供给、效果与评分目录，生成D/C |
| `stella/note_kernel.py` | NoteConfig／NoteLatent；initial、battle、random_gain、spend、apply_gain、score_notes、按来源计数抽类 |
| `stella/note_environment.py` | NoteEpisode／NotePublicState；复用共享潜能转移，保存商品、事件、来源和公开belief |
| `tools/generate_note_games.py` | 完整目录输入、合法策略、公开trace、完整checkpoint及续跑 |

NoteSetup字段为tower、main_disc_ids、auxiliary_discs、lead_element、initial_notes。默认塔light_earth，主秘纹214066／214039／214003，辅助[[214009,8],[214024,8]]，lead_element=null，initial_notes=null。initial_notes给出时替代辅助供给，仍单独发放一次开局包。已包含开局的续局快照使用opening_issued，禁止重复发放。

NoteLatent包含surprise_used、surprise_failures、boss_need_recovery、bonus_budget、early_extra_floor、middle_extra_floor、late_extra_floors、type_draw_counts、supply_issued、opening_issued及reward_context。类别计数按来源与子池分开，快照保存每个实际抽类的累计次数。

初始化与随机数消费顺序固定为：

1. 建立D/C/N、全部类别计数0、惊喜计数0和首领恢复标志false。
2. 抽共享T；依次抽有名额的早期位置、中段位置，再抽末段第一个、第二个位置。
3. 载入initial_notes或发放确定性供给，设置supply_issued。
4. 生成开局包：来源B选子池、按计数选类型、更新该计数、执行惊喜，再更新库存及opening_issued。
5. 按共享路线逐条推进。每个战斗包均按“子池→类型→计数更新→惊喜→库存”的顺序处理；正常第18层战斗的全部奖励处理完成后，按§5.5规范化惊喜状态为(4,0)。该操作不额外消费随机数。

`NoteEpisode(preset, seed, setup=None, config=None)` 初始化后自动推进无需决策的路线指令。公开state包含notes、setup、goods／full_stock、event_disabled、note_history、缺失掩码、公开belief及共享潜能字段；`state.to_dict()`输出公开JSON。ShopGood保留sid、type、note_id、quantity、base_price、price、role、货架generation及cause。

`episode.actions`列合法动作；`episode.step(action)`返回potential_effective／note_build／concerto_build三分项delta；`episode.score_components`返回当前分项；`episode.done`表示END。BUY_NOTE的index=sid+1、type=note_id、price=实际价。生成器EVENT_OPTION.index为静态optionId，实机提交端按当前显示菜单转为零基索引。

`episode.clone()`复制完整状态及游戏RNG；`snapshot()`与`NoteEpisode.from_snapshot(data)`精确续跑。`state_from_dict`只载公开观察，按公开来源与历史建立后验，不把公开库存直接当成隐藏计数。构筑修改按§8.2作为单独任务动作；正常爬塔执行到最终店END，不引入战斗伤害模拟。

`play(preset, seed, setup=None, config=None, policy=None, on_step=None)`默认均匀选合法动作。policy签名为(state, actions, rng)，使用独立planning_rng。公开trace保存before／after、合法动作、所选动作、分项delta与原始目标向量；私有checkpoint另存episode及policy RNG，不作为策略输入。

目标数据schema为 `complete_note_game_v4`；JSONL为 `joint_note_generation_v4`；checkpoint为 `joint_note_generator_checkpoint_v4`。每条记录同时保存model_version=`note_state_v5_20261115_r3`。CLI接口提供episodes、seed、trace、output、stop-after、snapshot、resume、catalogue-start、catalogue-id、preset、setup、profile、tower、policy。续跑采用checkpoint内的完整预设与配置。

实机记录与合成记录使用不同来源标记。实机来源以统一数据库中的物理局ID、record_id、原始库存和操作顺序定位；合成记录保存模型、配置、seed及完整内部状态，不写回实机记录。

### 9.1 服务器字段

| 消息 | 字段号及内容 |
|---|---|
| SubNoteSkillInfo | 1 Tid:uint32；2 Qty:int32；3 New:bool；4 LuckyLevel:uint32 |
| TowerChangeData | 1 Infos[]；2 Secondaries[] |
| ActiveSecondaryChange | 1 SecondaryId；2 Active |
| BattleCaseData | 1 TimeLimit；2 FateCard；3 SubNoteSkillNum |
| SelectOptionsEventCaseData | 1 EvtId；2 Options[]；3 FailedIdxes[]；4 NPCId；5 Done；6 Infos[]；7 EventOptionCnt:uint32 |
| HawkerGoods | 1 Sid；2 Idx；3 CharPos；4 Price；5 Discount；6 Type；7 GoodsId；15 Tag |
| HawkerCaseData | 1 List[]；2 Purchase[]；7 CanReRoll；8 ReRollPrice；9 ReRollTimes |
| InteractHawkerReq | 1 Sid；2 ReRoll；3 Auto |
| InteractSelectReq | 1 Index；2 ReRoll；3 Auto |
| InteractSelectResp.Success | 1 Items[]；2 OptionsResult；3 FateCard[]；4 SubNoteSkills[]；5 OptionsParamId；6 AffinityChange[] |
| StarTowerBuildDetail | 1 Potentials[]；2 SubNoteSkills[]；3 ActiveSecondaryIds[]；4 ModifierState |
| StarTowerBuildNoteModifyReq | 1 BuildId；2 NoteChange |
| StarTowerBuildNoteModifyResp | 1 Change；2 Info |

ItemTpl={Tid:uint32,Qty:int32}，原协议目录收录41种相关消息；本节补入当前国服2.1.0／资源2611131417的EventOptionCnt字段。10502在计数cnt=0／1／2时的实际成功率固定为30%／65%／100%，作为已定硬规则，不列待验，不继续采样或校准。字段来源与客户端显示消费保持明确：BaseRoom读取EventOptionCnt并作为面板第16参数传给NpcOptionCtrl，10502概率文案按30＋35×cnt及其补数100−(30＋35×cnt)显示。该字段不是音符隐藏类别计数或惊喜失败计数。Data.Infos／LuckyLevel及预览条目未提供时保留缺失，不由库存差补造协议值。[E8、E21、E30]

## 10. 静态目录与协议引用

运行需要完整NoteRules、效果目录和协议目录。以下相对路径以项目的ReverseEngineering还原源码／产物目录为基准；E27–E30定位当前国服客户端2.1.0、资源2611131417中的静态表、协议及消费方法。元素与单一通用类型等权已归入实测支持S17；12002各1/2分支与10502的30%／65%／100%实际成功率均为已定硬规则。其余引用保留原快照范围。概率程序的证据、分辨率和观察事项见[音符观察报告](NoteStateSpace_Observation_Targets.md)。

|编号|证据定位|内容|
|---|---|---|
|E1|`Game/Adventure/StarTower/StarTowerLevelData.lua` 316–469、708–759、1512–1543|快照、交互、增量和A_server|
|E2|同文件1713–1784、1808–1844、1904起、2407–2428|效果、预览、评分和C#同步|
|E3|`Game/Adventure/StarTower/StarTowerRoom/BaseRoom.lua` 778–917、1066–1117|当前事件与商店|
|E5|`Game/UI/StarTower/StarTowerShop/StarTowerShopCtrl.lua`483–496、542–567；GridCtrl28–101；StarTowerAutoDecision|购买、实时价、刷新和类型|
|E6|`Game/UI/BuildModifier/Note/BuildModifierNoteCtrl.lua`56、99–106、225–254；ListGridCtrl78–83|修改额度与差值|
|E7|`Game/UI/StarTower/Build/StarTowerBuildDetailCtrl.lua`193–224；PlayerBuildData:ModifyBuildNote|修改入口和构筑响应|
|E8|`_unnamed/1a70da105b39911e.luac`|原快照357份proto的FileDescriptorSet；当前字段补充见E30|
|E9|`artifacts/tables/SubNoteSkill.bytes`、SecondarySkill、Disc、配置和掉落分组|静态数值；完整偏移与摘要见JSON|
|E12|`GameCore/Data/DataClass/DiscData.lua`83–146、331–383、397–557；PlayerDiscData50起、93起、360起；`Game/UI/MainlineFormationDiscEx/MainlineFormationDiscCtrl.lua`137–151、241–255|秘纹算法、主辅供给和终局采用A_server|
|E13|`Game/CodeGen/GAME_ENUM_DEFINE.lua`2693–2711；BaseRoom827–880；NpcOptionCtrl155–292、679–686；事件表及中文文本|事件枚举、效果、菜单与失败|
|E17|`work/effect_CommonHelper.cs`131–139、1494–1524；ActorEffectManage453–456；AttributeEntry19；`data/stella_note_effects_verified.json`|真实参数、效果查表、Lucky显示及100／101边界|
|E18|`artifacts/decompiled/StarTowerLevelController.cs`82–160、1194–1241、1306–1321、1381–1399、1459–1490、1617–1672；`lua_drop_input.txt`|音符掉落表现链|
|E19|`artifacts/decompiled/AdventureModuleHelper.decompiled.cs`2074起；AdventureModuleLevelLoadParams421–630|本地地图选择|
|E20|`artifacts/decompiled/TrueSync.FDP.native.asm`、`.metadata.txt`、`.decimal_calls.txt`|signed64、24位小数、decimal四舍五入和饱和|
|E21|`data/note_random_cil_references.json`、`note_random_memberref_aliases.json`；`reports/note_random_hotfix_audit.txt`|全部CIL直接调用审计|
|E22|`artifacts/decompiled/GameController.Random.native.asm`10、82、139、332、519、551|本地RNG初始化、右开边界和取模|
|E24|`scripts/current/export_note_rules.py`；`data/NoteRules.json`源记录摘要与discs；`data/table_index.json`的Disc／TrialDisc schema；`work/{cn,jp}_unreleased`完整表|106件秘纹目录、阶段供给、协奏组、可见／可用标记和试用字段映射|
|E26|`artifacts/tables/StarTowerGrowthNode_a5314bacab7c685b.bytes`及日文`_ab4b1ebea19b769d.bytes`；`work/stella_lucky_ui.lua`43–62、`stella_lucky_active.lua`151–159及188–189、`stella_lucky_item.lua`29–38|研究节点文本、UI缓存差与LuckyLevel布尔展示|
|E27|当前CN `C:/YostarGames/StellaSora_CN/Persistent_Store/Tables/data.arcx`：SubNoteSkill[90011–90023]；完整45条Character.EET；StarTower[109/209/309].EET及SubNoteSkillDropGroupId；SubNoteSkillDropGroup组2/3/4；StarTowerGrowthNode[40101].IsServer/EffectClient/Clientlvl及中文Desc|13类、角色元素、三塔原始掉落目录与40101设计条件；162/115/107的EET均为5。元素与单一通用等权的实测结论另见观察报告S17|
|E28|同一当前CN归档：EventOptions[12002]，记录偏移0x915及其中文Desc；EventResult[1200201]偏移0x5f1、[1200202]偏移0x607；当前GAME_ENUM_DEFINE的RecoveryHP=14、GetRandomSubNodeSkill=17|12002各1/2恢复30%生命／随机5枚的确定分支规则及两结果配置；第一结果另含Effect2=16、Parameter2=[1110101]|
|E29|当前CN `Persistent_Store/Scripts/lua.arcx`：StarTowerShopCtrl:OnBtn_Confirm 569–582；StarTowerLevelData:ProcessTowerChangeData 1495–1524、评分段1912–1920、ResetNoteInfo 2411–2419；当前Config.DiscSubNoteSkillMaxLevel及SubNoteSkill.Scores|当前数量≥99禁购、评分按99项封顶、库存按Qty累加及原值同步；本体固定奖励库存允许超过99，不截断或拒收|
|E30|同一当前CN lua.arcx，descriptor fid `1a70da105b39911e`、SHA256 `3bc6f7b10a694a7d6fdd2db0687cc080f7b3dbaf675958dbe42ece9e168546ad`；BaseRoom 945–946；NpcOptionCtrl 166、273–276|SelectOptionsEventCaseData字段7 EventOptionCnt:uint32；面板第16参数及10502的30＋35×cnt概率文案消费。10502在cnt=0／1／2时实际成功率已固定为30%／65%／100%，不列待验；该字段不作音符隐藏计数|

## 11. 版本与快照契约

本体版本固定为 `note_state_v5_20261115_r3`。初始化、动作转移、抽类计数、阶段位置与随机流均随版本记录。快照仅由相同版本恢复；版本不一致时拒绝直接续跑。公开历史导入按本版本观测核建立belief，不给隐藏预算、包顺序或抽类计数强配单一值。经验证的正常第18层完整前缀按§5.5合并惊喜状态；其余隐藏维度继续按相容路径保存。

从 `note_state_v4_20261115_r2` 迁移时，生成概率、状态坐标和随机数消费顺序保持分布等价。迁移器先核对旧快照的完整规则配置、目录、来源、开局标志、路线位置及完整状态或联合后验，确认其与本版本转移核相容；通过校验后复制全部状态、随机流与后验，并登记新版本。已验证的正常第18层完整前缀仅合并惊喜坐标为(4,0)，对被合并路径的概率求和，保留类别计数、首领记忆等其他隐藏维度及其联合相关性。其余位置保持旧后验，不靠楼层标签代替前缀验证。

旧观察历史或后验在本版本下没有相容路径时，拒绝等价迁移并保留原快照；不得通过强制写入(4,0)修复不相容历史，也不得只修改版本字符串直接续跑。需要改变转移核的更早版本按其完整公开历史重放观察更新后，再生成本版本快照。

本体规定唯一的执行程序；证据与观察事项见[音符状态空间观察报告](NoteStateSpace_Observation_Targets.md)。
