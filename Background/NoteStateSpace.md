# 音符状态空间

版本：2026-11-08。客户端资料：本次本地日服客户端快照。共享路线与潜能规则见[潜能状态空间](StateSpace.md)。

**默认规则：音符始终支付得起。** 支付5枚或10枚音符的选项按可支付处理，支付后按实际选中的音符种类与数量更新库存。

## 1. 规则级别

|级别|内容|
|---|---|
|确定规则|音符足额支付的默认设置；库存更新、辅助秘纹供给、协奏激活、评分、效果参数、事件配置、每架8件商品及构筑修改规则|
|实机数据高置信度支持的规则|每架4件折扣、半价最多2件、大包最多1件；大包出现率约80%；107／108的可见菜单结构|
|待确定的规则|战斗获得音符的数量与种类概率、幸运图标对应的数量变化、兑换时选走哪些音符、菜单组合概率、商品与折扣的联合分布|

## 2. 固定输入与状态

### 2.1 固定输入

- 客户端／资源版本、塔ID（`StarTowerId`）、路线与研究目标。
- 队伍、角色位置，以及主辅秘纹的ID、等级、阶段`Phase`和星数`Star`。
- 共享潜能模型的预设与计分规则。

本塔的音符候选集合由`SubNoteSkillDropGroupId`指定。[E1]

|组|音符ID集合|
|---|---|
|1|90011–90017|
|2|90011–90017、90021、90022|
|3|90011–90017、90018、90020|
|4|90011–90017、90019、90023|

### 2.2 动态状态

\[
S=(f,q,\mathbf n,c,m,h,P,u,A_{server},C,H,E,B).
\]

|分量|含义|
|---|---|
|`f,q`|楼层、阶段、房型及当前操作阶段|
|`n`|90011–90023全部13种音符的实际持有数量|
|`c`|星塔币|
|`m`|相关材料数量|
|`h`|当前生命、最大生命及相关效果|
|`P`|当前潜能持有与流程状态，使用潜能状态空间的完整定义|
|`u`|天赋、成长节点和当前生效的奖励修正条件|
|`A_server`|服务器返回的已激活协奏ID|
|`C`|当前交互对象的ID、类型、数据和完成状态|
|`H`|店序、初始／刷新、货架身份、完整商品及后台顺序、已购商品、刷新许可／费用／剩余次数|
|`E`|事件类型、NPC、题目、可见选项及顺序、禁用项、完成状态、本局已出现事件|
|`B`|构筑ID、锁定／试用状态、分数、修改状态和修改前音符数量|

每件商品记录类型、音符种类、包数量、原价、实际售价和对应角色。每次操作记录对局ID、动作ID、顺序、数据来源和读取完整性。观测缺失记为`null`，明确读到的零记为0。

### 2.3 派生状态

给定音符数量和主秘纹，本地计算：

- 每组最高激活协奏`A_local`。
- 音符构筑分、协奏构筑分。
- 音符与秘纹提供的战斗效果输入。

`A_local`随库存重算；`A_server`按服务器响应更新。[E1、E2、E12]

## 3. 确定规则

### 3.1 库存累计与辅助供给

进塔／续局使用`Bag.Items`中的音符快照赋值；交互后按`Data.Infos`逐条累加：[E1]

\[
n_{Tid}\leftarrow Qty\quad\text{（快照）},\qquad
n_{Tid}\leftarrow n_{Tid}+Qty\quad\text{（变化量）}.
\]

例如原有10枚攻击音符，收到`Qty=+5`后为15枚，收到`Qty=-5`后为5枚。

辅助供给共有116组、每组`Phase=0–8`，合计1044条。按辅助秘纹的`SubNoteSkillGroupId`和阶段查找当前阶段及以下最近一条记录，直接采用该记录；各辅助秘纹的供给相加。起始快照包含供给时，直接使用该快照。[E12]

### 3.2 音符评分与协奏

音符构筑分为：[E9]

\[
V_{note}=15\sum_{i=90011}^{90023}\min(n_i,99).
\]

751条协奏包含150组正常协奏的5级记录及特殊记录9001；316条需要两种音符，435条需要三种音符。常见等级门槛为10／25／40／55／70和15／30／45／60／75。

对前三个主秘纹的每组协奏，检查所有所需音符是否达到数量要求，选取满足要求的最高一级。协奏构筑分为各主秘纹所含协奏组最高激活级的分数之和。[E9、E12]

### 3.3 效果与数值

主技能按`MainSkillGroupId`与`Star+1`选行；协奏按主秘纹对应组和音符需求选行。音符效果以实际持有数量作为等级参数，秘纹效果由主技能与各组最高激活协奏组成，挂载到全队。[E2、E12]

|音符|每级参数|属性处理|
|---|---:|---|
|90011 攻击、90014 生命|0.003×L|PERCENTAGE|
|90012 暴击率|0.001×L|BASE_VALUE|
|90013 暴伤、90016 技能伤害|0.0046×L|BASE_VALUE|
|90015 普攻伤害|0.012×L|BASE_VALUE|
|90017 必杀伤害|0.01×L|BASE_VALUE|
|90018–90023 六元素伤害|0.003×L|BASE_VALUE|

属性合成为`(origin+baseAmend)×(1+percentAmend)+absAmend`。配置参数通过decimal转为FDP，FDP为有符号64位定点数，数值等于`raw/2^24`；decimal转换采用最近整数，中点远离零。完整1287条效果值及量化结果见[效果目录](../Research/data/stella_note_effects_verified.json)。[E17、E20]

库存增加且`LuckyLevel>0`时显示幸运图标，数量使用实际库存变化。首次载入库存时直接显示当前持有量。[E17]

### 3.4 事件配置与操作

默认音符支付始终满足，兑换时消耗的具体种类按服务器结果更新。

|选项|配置结果|
|---|---|
|10301|随机消耗5枚音符，获得150币|
|10401|随机消耗10枚音符，获得随机稀有潜能|
|10701–10707|分别获得90011–90017中对应种类5枚|
|10708|获得随机音符5枚|
|10801–10807|花140币，获得对应种类音符10枚|
|10808|花90币，获得随机音符10枚|
|11403|答对，获得随机音符10枚|
|11401／02／04／05|答错，显示错误提示|
|12001|花50币，获得随机音符5枚|
|12002|配置为50%恢复30%生命、50%获得随机音符5枚|
|12702|获得随机音符5枚|

客户端读取服务器的`Options`及顺序，提交零基显示索引；成功后标记完成，错误答案等失败结果更新禁用项`FailedIdxes`。[E3、E13]

事件出现后加入本局已见集合，固定NPC和追加NPC共用去重。事件类型、路线与共享事件概率使用潜能状态空间第10节；音符模型保留完整菜单和具体音符结果。

### 3.5 商店

每次生成完整货架固定8件商品。购买按商品流水号`Sid`提交，实际售价为`Discount>0 ? Discount : Price`。当前商品类型1为潜能，类型2为音符。[E3、E5、E8]

共享研究设置的刷新费为100币，刷新后剩余次数减1并替换整架。实际对局记录响应中的许可、费用和次数。

### 3.6 构筑修改

入口条件：非限时试用、未锁定、构筑分数达到5800、修改状态为0或1，消耗材料35013×1。状态2为已改音符，状态3为不可修改。[E6、E7、E9]

设提交的变化为\(\Delta\)，可调换1–15枚：

\[
\sum_i\max(\Delta_i,0)=\sum_i\max(-\Delta_i,0)=k,
\qquad 0<k\le15,\qquad 0\le n_i+\Delta_i\le99.
\]

请求提交新减旧的差值，成功后使用返回构筑。终局效果按返回的激活协奏ID加载。[E7、E12]

音符评分上限、商店当前数量检查和构筑修改界面使用配置99。[E5、E6、E9]

### 3.7 战斗表现与地图

战斗音符动画使用服务器预览数量作为表现额度，按怪物类型分配掉落动画，结束时补播剩余额度；实际音符库存按交互响应更新。[E18]

地图选择在本地按路线、楼层配置和历史地图筛选候选，再调用本地RNG。固定路线使用潜能状态空间的路线定义。[E19、E22]

## 4. 实机数据高置信度支持的规则

数据为401040条总记录，分析得到1809架货架、315个对局分组、18名货架数据参与者。[E23]

|规则|实机结果|
|---|---|
|每架恰4件商品打折|1809／1809架符合|
|每架半价商品最多2件|1809／1809架符合；0／1／2件分别为50／298／1461架|
|每架15枚音符大包最多1件|1809／1809架符合；0／1个分别为358／1451架|
|大包整架出现率约80%|1451／1809＝80.21%|
|107菜单含两个不同指定音符项及随机音符项|41／41例符合|
|108菜单含两个不同指定音符项、随机音符项及30币项|32／32例符合|

大包出现率的对局分组bootstrap 95%区间为78.38%–82.01%，参与者分组区间为75.56%–81.08%。精确4/5及其条件关联列入待确定的规则。

实机统计反对“各音符商品格共用同一概率、独立决定5枚或15枚”的生成方法；每位参与者取一架的条件精确检验\(p=0.000183\)。统计方法、分组与价格归一规则见[实机统计报告](../Research/reports/NoteRealGameValidation.md)。

## 5. 服务器返回的数据

|场景|服务器返回|客户端处理|
|---|---|---|
|进塔／续局|各类音符数量、激活协奏及当前交互对象|载入当前状态|
|战斗／事件结束|音符种类、增减数量、幸运标记及其他资源变化|累加数量、重算本地协奏和分数、显示结果|
|事件菜单|事件、NPC、选项顺序、禁用项、完成状态、结果标识|显示实际菜单并提交选择|
|商店／刷新|完整商品、已购列表、刷新许可／费用／次数|显示货架、提交商品或刷新请求|
|构筑修改|修改后的构筑及激活协奏ID|采用返回构筑|

音符正常更新链为交互回调`Data.Infos`；库存、事件和货架的响应消费链见[E1、E3、E8、E21]。

### 5.1 关键协议字段

|消息|字段号及内容|
|---|---|
|`SubNoteSkillInfo`|1 Tid:uint32；2 Qty:int32；3 New:bool；4 LuckyLevel:uint32|
|`TowerChangeData`|1 Infos[]；2 Secondaries[]|
|`ActiveSecondaryChange`|1 SecondaryId；2 Active|
|`BattleCaseData`|1 TimeLimit；2 FateCard；3 SubNoteSkillNum|
|`SelectOptionsEventCaseData`|1 EvtId；2 Options[]；3 FailedIdxes[]；4 NPCId；5 Done；6 Infos[]|
|`HawkerGoods`|1 Sid；2 Idx；3 CharPos；4 Price；5 Discount；6 Type；7 GoodsId；15 Tag|
|`HawkerCaseData`|1 List[]；2 Purchase[]；7 CanReRoll；8 ReRollPrice；9 ReRollTimes|
|`InteractHawkerReq`|1 Sid；2 ReRoll；3 Auto|
|`InteractSelectReq`|1 Index；2 ReRoll；3 Auto|
|`InteractSelectResp.Success`|1 Items[]；2 OptionsResult；3 FateCard[]；4 SubNoteSkills[]；5 OptionsParamId；6 AffinityChange[]|
|`StarTowerBuildDetail`|1 Potentials[]；2 SubNoteSkills[]；3 ActiveSecondaryIds[]；4 ModifierState|
|`StarTowerBuildNoteModifyReq`|1 BuildId；2 NoteChange|
|`StarTowerBuildNoteModifyResp`|1 Change；2 Info|

`ItemTpl={Tid:uint32,Qty:int32}`。`StarTowerInteractResp`包含Cases、Change、Data、EnterResp、BattleEndResp、SelectResp、StrengthenMachineResp和Settle等结果。完整41种相关消息见[协议字段表](../Research/data/NoteProtocol.json)。[E8]

## 6. 完整目录与资料

- [规则目录](../Research/data/NoteRules.json)：13种音符、751条协奏、104件秘纹、1044条阶段供给。
- [共享塔与事件配置](../Research/data/TowerRuleTables.json)：相关配置键、事件选项与结果、塔／阶段、音符池和构筑配置。
- [秘纹目录对照](../Research/reports/DiscCatalogCoverage.md)：本机104件与本次核对的日服已实装图鉴104件逐项对应；61件双协奏、26件单协奏、17件无协奏。[E24、E25]
- [效果目录](../Research/data/stella_note_effects_verified.json)：13×99条效果值与FDP量化结果。
- [协议字段表](../Research/data/NoteProtocol.json)：本次客户端的41种相关消息与字段。
- [补录要求](../Research/data/NoteValidationCapturePlan.json)：三类规则和后续记录字段。
- [研究资料与分析入口](../Research/README.md)：公开配置目录、实机派生统计和分析命令。

## 7. 资料来源索引

客户端行为条目来自本次客户端资料的规则整理，底层实现文件不作为仓库附件。公开配置目录保留表名、来源SHA256及必要记录定位；协议字段、实机统计和线上目录对照分别列出来源。

|编号|来源与公开整理目录|内容|
|---|---|---|
|E1|客户端状态同步行为整理；[协议字段表](../Research/data/NoteProtocol.json)|库存快照、交互增量和服务器激活协奏|
|E2|客户端效果与评分行为整理；[规则目录](../Research/data/NoteRules.json)、[效果目录](../Research/data/stella_note_effects_verified.json)|效果输入、本地协奏、评分和战斗同步|
|E3|客户端事件／商店响应处理整理；[协议字段表](../Research/data/NoteProtocol.json)|当前事件、可见菜单、商品和操作请求|
|E5|客户端商店行为整理；[协议字段表](../Research/data/NoteProtocol.json)、[塔配置](../Research/data/TowerRuleTables.json)|购买、实时价、刷新和商品类型|
|E6|客户端构筑修改行为整理；[塔配置](../Research/data/TowerRuleTables.json)中的Config|修改额度、数量上限和提交差值|
|E7|客户端构筑入口与响应处理整理；[协议字段表](../Research/data/NoteProtocol.json)、[塔配置](../Research/data/TowerRuleTables.json)|修改入口、材料和返回构筑|
|E8|[协议字段表](../Research/data/NoteProtocol.json)所记客户端字段描述SHA256与357份文件定义|41种相关消息的字段、类型及嵌套关系|
|E9|[规则目录](../Research/data/NoteRules.json)、[塔配置](../Research/data/TowerRuleTables.json)|音符、协奏、秘纹、供给、配置和掉落候选分组|
|E12|客户端秘纹数据处理整理；[规则目录](../Research/data/NoteRules.json)|主技能选行、最高激活协奏、辅助供给和终局激活结果|
|E13|客户端事件菜单行为整理；[塔配置](../Research/data/TowerRuleTables.json)中的事件表|事件效果、菜单顺序、失败与禁用项|
|E17|[效果目录](../Research/data/stella_note_effects_verified.json)|真实参数、效果查表、幸运图标显示及100／101边界|
|E18|客户端战斗表现行为整理，见本文第3.7节|服务器预览数量、掉落动画与实际库存分别处理|
|E19|客户端地图选择行为整理，见本文第3.7节；共享路线见[潜能状态空间](StateSpace.md)|本地地图候选与路线条件|
|E20|[效果目录](../Research/data/stella_note_effects_verified.json)中的arithmetic|signed64、24位小数、decimal转换和饱和|
|E21|客户端交互响应使用关系整理；[协议字段表](../Research/data/NoteProtocol.json)|音符、事件、货架读取服务器返回结果|
|E22|客户端本地随机接口行为整理|本地RNG初始化、右开边界和取模；不作为服务器音符生成概率证据|
|E23|[实机统计报告](../Research/reports/NoteRealGameValidation.md)、[商店统计](../Research/data/realgame_shop_stats.json)、[事件统计](../Research/data/realgame_events_analysis.json)、[覆盖统计](../Research/data/realgame_notes_coverage.json)|401040条记录、1809货架、菜单实例、分组区间和条件精确检验；输入为唯一实机总数据|
|E24|[规则目录](../Research/data/NoteRules.json)的完整discs与来源摘要；[目录对照](../Research/reports/DiscCatalogCoverage.md)|104条秘纹定义、61／26／17协奏分类|
|E25|[目录对照](../Research/reports/DiscCatalogCoverage.md)、[逐项匹配](../Research/data/online_disc_catalog_comparison.json)及其中公开来源链接|线上104条与本地104条逐项匹配；101直接对应、3处用字／符号差异|

## 8. 待确定的规则与实机记录

|待确定的规则|需要记录的数据|
|---|---|
|每场战斗获得多少音符、各类如何分配、队伍／秘纹／房型的影响，以及8000和权重配置的实际作用|单场战前／后13种音符数量、数量预览、胜负、来源、队伍与主辅秘纹、期间其他音符变化|
|幸运图标出现概率、等级含义及对应增加数量|全部获得音符的机会、是否出现图标、实际等级、每类增加数及可知基础量|
|用5枚换150币／用10枚换稀有潜能时，各类音符怎样被选走|兑换前／后13种数量、选择及实际结果、币／潜能变化；保留种类数量不均衡的样本|
|事件菜单组合概率、NPC／版本条件、114题目与正确选项对应、12002实际分支频率|完整菜单及顺序、选项ID、NPC／题目、本局已见事件、所选项与结果；回血分支同时记录生命／最大生命前后|
|大包精确出现率、音符种类与包大小／饮料／折扣的联合分布、店序／刷新的影响|第几家店、楼层、初始或刷新、购买前全架、商品种类／数量／原价／实际价、后台列表顺序与刷新参数|

共享潜能规格第8节当前采用E6作为商店数量工作核；在音符研究中，该精确生成机制及竞争假说仍保留为待确定规则。实机支持的货架结构不等于已确认完整联合分布：

- 当前数量工作核：每架重置十张票，仅Y／B不放回取两票，阈值分别为6和8；X另按四次机会生成并截到2，Y=0时概率为0.65，否则为0.60。`G=2-b+X`、`M=b+Y`，首店初始与补货均取`b=1`，其余取0。完整定义及竞争假说见[共享商店规格](StateSpace.md#8-商店与货架)和[观察清单](StateSpace_Observation_Targets.md#当前商店模型与竞争假说)。
- 大包精确出现率4/5，以及包大小、音符种类与其他商品数量的条件关系。
- 在8件商品中均匀选4件折扣；半价数`D=min(2,Binomial(3,p))`，第2、3店初始架`p=3/4`，其余初始架与刷新`p=7/10`。
- 菜单中21种无序音符对等概率出现。

单场战斗、幸运标记和兑换结果目前可对应的完整音符前后数量记录均为0。下一批数据按动作关联前后状态，保留`Data.Infos`完整数组及其`Tid／Qty／New／LuckyLevel`；新记录并入`Background/RealGameData/data/observations.jsonl`。
