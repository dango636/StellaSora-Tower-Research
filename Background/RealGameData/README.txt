Stella Sora 星塔实机研究数据

从 GitHub 仓库的 real-game-data Release 下载并解压。总库在 Background/RealGameData/database/research.sqlite3，可用 SQLite 打开。数据库不放在 Git 提交里。包内 SUMMARY.json 是这份数据库的实际计数；Git 仓库里的 SUMMARY.json 是最近整理研究资料时的计数。旧版 observations.jsonl 已并入 SQLite，不再另行分发。

数据如何关联
research_schema.sql 定义 8 张表和 15 个查询视图。runs 每行是一个采集段，包含区服、预设、开始和结束时间、结束原因及质量；presets 保存原始目标向量。records 记录游戏动作与观测，objects 和 readings 分别保存实体及带位置、置信度的读数，通过 record_objects 和 record_readings 与观测相连。quality_issues 定位有问题的对局、记录或字段。
总库只收录有可靠物理开局或已确认接续归属的采集段；无开局依据、未归属及仅元数据段不入库，实机来源标记不能作为例外。legacy exploration_id 是明确开局，即使原 full_run_start_observed=0 仍保留。按已确认的 config_json.physical_game_id 分组；无此字段时用 _identity.anchor_run_id，再没有则用 run_id。同一对局可能跨多个采集段，其中任一已确认归属的段观测到结算，才计一次完成。中途接入且读到结算不等于新增完整对局。SUMMARY.json 分别列出实际对局、完成局、采集段及一局多段的数量。
预设的 36 项原始目标不计核心，分别对应三名角色各 3 个稀有和 9 个普通潜能。已知未选为 0，未知为 null。target_vector_complete 只说明这 36 项齐全。metadata/potentials.json 提供潜能身份、区服名称和品质；Git 中 metadata/presets.json 是旧采集格式使用的预设字典，完整预设以数据库的 presets 表为准。Release 包内的预设目录从该次数据库生成。

如何用于分析
使用 record_quality 和相关视图（例如 potential_candidates、shop_slots、note_inventory、note_deltas、settlement_ui_totals）取出要分析的记录。不要把同一画面的重读当作新的抽取。保留冲突和未知；原始 OCR 表示程序读到了什么，不自动代表游戏真值。
普通战斗按本塔画面显示的九种音符（七通用、两属性）核对变化，其余四种不补零。研究随机扣除规律时另取十三种前后库存均已知、数量及来源已确认的动作。中断对局仍可提供已独立确认的区间；无法拆开的混合区间不能冒充单次战斗或单次事件。
训练资格、概率资格和终局是否完成含义不同；本库不提供教师标签。不能根据识别成功、预设完整或一次局部审核自行提升资格。公开库将参与者与会话标识换为稳定代号，保留研究关联，去除本机路径、已识别的账号信息、截图和原始调试日志。

从研究仓库根目录使用工具（Python 标准库）
按动作审核音符区间，仅输出汇总：
  python Research/analysis/note_quality.py Background/RealGameData/database/research.sqlite3
将指定原生对局文件合入自己的数据库，相同材料不会重复计数：
  python Research/analysis/import_runs.py --database combined.sqlite3 run.json
也可指定原生单局 ZIP 或 SQLite。详见 Research/analysis/NOTE_QUALITY.md 和 IMPORT_RUNS.md。
