# 原生对局合并

`import_runs.py` 只读取命令中指定的 Maa 原生 JSON、GZIP、上传 ZIP 或八表 SQLite 文件，合并到指定数据库。只依赖 Python 标准库，默认使用 [research_schema.sql](../../Background/RealGameData/research_schema.sql)。

从仓库根目录运行：

```powershell
python Research/analysis/import_runs.py --database /path/to/combined.sqlite3 /path/to/run.json
python Research/analysis/import_runs.py --database /path/to/combined.sqlite3 /path/to/run.zip /path/to/another-run.zip
python Research/analysis/import_runs.py --database /path/to/combined.sqlite3 /path/to/native.sqlite3
```

JSON 使用 UTF-8（可带 BOM），格式为 `maa-state-space-database`、`schema_version=1`。上传 ZIP 包含 `research.json` 或 `research.json.gz`，支持嵌套 ZIP；文件不解压至磁盘。SQLite 输入只读，读取该文件中的全部对局。可用 `--schema /path/to/research_schema.sql` 明确指定同源正式结构。

多个输入按参数顺序、ZIP 按成员顺序处理。已有可信主值优先，后续来源只补齐缺失内容；真实冲突保存双方原文、关联和未确认状态。同包重复导入不会新增对局、记录或读数；同包改名只补来源。`target_vector_complete` 检查完整原始36维向量，不推断训练资格。

程序逐局提交，输出 JSON 批次统计；中断后可重跑原命令。目标库初始化会原子对齐正式八表、十五视图，并移除旧训练范围列和视图，不重写观测表。它不会查找其他文件、回填历史、重新划分物理对局、清理总库或更新 `SUMMARY.json`。音符统计审核继续使用 [note_quality.py](note_quality.py)，口径见[审核说明](NOTE_QUALITY.md)。

维护源位于 MaaStellaSora 的 `tools/data/real_game_merge.py` 与 `real_game_selected.py`，此处两文件保持同源一致。公开入口不依赖 Maa、旧 reader、训练器或任何本机路径。内存调用可使用 `import_runs.import_payloads(connection, [(label, payload)])`。
