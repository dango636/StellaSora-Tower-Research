Stella Sora 实机总数据

共 475,973 条记录、26,309 条候选观测、420 个对局标识、23 套预设。
观测时间：2026-10-21T01:49:36.762+09:00 至 2026-11-08T22:04:44.557+09:00。

文件
data/observations.jsonl：唯一总数据，UTF-8，每行一个 JSON 对象，按 record_id 去重。
metadata/presets.json：以 preset_id 关联的完整预设定义，角色位置统一为1/2/3；1为主角色。
metadata/potentials.json：以 potential_id 关联的潜能字典，合并中日文名称、名称检索键及核心/稀有/普通分类。
SUMMARY.json：总数据的当前计数，不保存导入或版本历史。

数据格式
schema_version=1 表示本总数据格式；event_type 区分候选、选择、刷新、商店、事件、钱包、结算及对应观测记录。
record_id 是唯一记录标识；participant_id、session_id、run_id 用于分组；timestamp 是实际观测时间，统一为日本时区 +09:00。
lang_type 使用 jp/cn/tw，分别表示日服、国服、繁中服；preset_id 使用 P0001 等稳定键，未知为 null。
预设定义中每个角色的 targets 按 potential_id 排列；target_vector 不包含核心，按角色1/2/3、各角色稀有3项及普通9项、同类潜能ID升序排列。
target_level=0 表示未选，null 表示未知。未保存具体核心ID的预设保留 null，并在已知时保留核心目标数量。
候选与持有项的 rarity_class、target_level、is_preset_target 已内联，不再依赖独立注释文件。
rarity_class 按潜能字典分为 core/rare/common/unknown；原识别 rarity 与 raw_rarity 保留其观测含义。
核心为 Item.Stype=42；非核心 Stype=41 中 Rarity=1 为稀有、Rarity=2 为普通。

使用口径
数据按输入对局顺序合并，不保证全局时间有序；按 participant_id/session_id/run_id 分组，使用 run_sequence（存在时）或 timestamp 排序。
保留实际游戏观测、OCR、动作关联、识别错误与完整性字段；未知身份不强行补值。分析候选时结合 candidates_complete、recognition_status 等字段筛选。
有效数值字段中的 null 表示未知或不可用，不能填成0、沿用上次值或按模型期望补齐；钱包识别失败的-1不作为金币值使用。
probability_eligible=false 的记录不进入概率统计的分子、分母；training_eligible=false 的记录不作为训练样本或标签。invalid_fields 标明失效字段；source_complete=false 表示奖励来源未建立关联，候选识别内容仍保留。两字段未出现时，仍须结合原有完整性、确认状态及目标向量要求筛选。
sequence_inference_eligible=false 或 sequence_complete=false 表示页面身份或观测序列不完整，不能用于重建h/P/Q、池顺序、奖励次数和依赖完整历史的训练样本；不得跨过缺失段继续累计。
invalid_fields 使用JSON路径标出不可用字段；recognition_status=invalid 的识别结果不可用，wallet_complete=false 时不能用该条记录推导钱包差额。不可用的有效数值直接置null。
raw_ocr、raw_result、raw_*_recognition保留原始识别证据，不是游戏真值或训练标签；使用有效价格、有效钱包和已解析身份字段。Maa_success、Maa_reported_success仅为软件回报，operation_confirmed=false时不能计为已发生的游戏操作。
重复观察、刷新和同一页面重读仍是实际记录，不能把每条候选观测直接当作独立抽取。
持有记录区分观测重建与独立验证；不同预设及来源条件按各自字段分组。
实机记录的预设范围与训练范围分别管理；target_vector_complete 仅表示目标向量齐全，不表示通过训练范围验证。training_scope_match 按原始完整目标向量判定；范围外的新记录保留用于实机分析，并标记 training_eligible=false。
已知商店价格修正直接体现在有效价格中；总数据只保留修正后的单一结果。
后续样本直接合并到上述文件并更新当前计数，不保留收包日志、修复日志、版本副本、来源归档、打包清单或校验历史。
