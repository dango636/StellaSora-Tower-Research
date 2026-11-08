# Stella Sora 训练器

本目录提供可编辑的模拟、神经网络训练和本地监控源码。游戏规格见仓库根目录的 `Background/StateSpace.md`；学习范围的机器约束见 `training/value_model/preset_scope.json`。

## 环境与启动

使用 Python 3.10 或更新版本，以及可用 CUDA 的 PyTorch 2.7 或更新版本。安装与设备匹配的 PyTorch、NVIDIA 驱动和 NVRTC 后，在仓库根目录安装其余依赖并启动：

```powershell
python -m pip install -r NeuralNetwork/trainer/requirements.txt
python NeuralNetwork/trainer/launcher.py
```

Windows 也可双击 `NeuralNetwork/启动训练.cmd`。入口使用当前环境中的 `python`，启动后会打开本地进度页面。RTX 5070 可增加 `--profile local_5070`；96 GB Blackwell 环境可增加 `--profile colab_blackwell96`。

默认运行目录为 `NeuralNetwork/trainer/runs/catalogue4960`。另建独立运行可指定目录：

```powershell
python NeuralNetwork/trainer/launcher.py --run-dir NeuralNetwork/trainer/runs/my_run
```

首次启动读取 `config.json`，在运行目录保存配置，并从仓库根目录 `models/initial.npz` 创建本次运行的 `M000000`。全部 4960 个预设最初使用这一模型作为教师，学生也从这一模型开始。首次启动会完成 4960 个预设、每预设 256 局的教师基线评测，然后进入第一轮训练。该基线及之后的训练过程均可保存和恢复。

再次使用相同命令会恢复该运行目录。进度页面可暂停、继续或保存并停止训练；关闭页面后训练继续运行。不同运行目录各自维护模型和教师映射。

## 当前训练与晋升协议

- 学习范围为 4960 个等权完整结构：每角色选择 3–9 个潜能，目标等级为 0/1/3/6；角色目标级数 18–54、全队 84–132，不计核心。完整约束和规范化方式以机器约束文件为准。
- 每轮每预设以独立轨迹产生 3 个训练根与 1 个自然验证根，共 19840 根；标签由轮初冻结的该预设教师完成动作续局生成。
- 使用 `action_gap_calibration_v3`，学习率 `1e-5`、`uncertainty_trust_weight=0`、`reliability_floor=0.25`。学生从上一轮正式候选初始化，按标签后悔值或动作优势 RMSE 的改善控制早停。
- 每轮保留 6 个实际训练检查点，优先保留 epoch 1/2/3/4、验证指标最佳点和最后实际 epoch，重复名额按 epoch 距离补足。
- `six_checkpoint_official_v1` 对每个候选评测全部 4960 个结构，每结构 256 局，共用固定正式评测种子。种子与采样、标注独立。
- 每个预设从 6 个候选中选择总分最高者，与轮初该预设教师比较；256 局整数总分至少增加 13 才更新教师。全局最佳单模型按完整目录等权均值严格提高时更新。
- 选模与教师晋升共用上述评测；区间仅作描述。所有策略动作直接由网络 argmax 决定，保持 FP32。

`config.json` 是可以直接用于新运行的完整当前配置；`profiles/training.json` 是相同训练设置的覆盖配置。更改已有运行的训练设置可使用 `tools/prepare_calibration_revision.py` 将覆盖配置应用到下一未冻结轮。模型、轮次状态、教师映射、训练标签和缓存均由程序在各运行目录生成。

## 源码与测试

`stella/` 包含模拟器、特征、网络、采样、训练和选模；`web/` 为本地监控页面。第三方来源见 `THIRD_PARTY_NOTICES.txt`、`stella/vendor/SOURCE.json` 和 `web/assets/`。

普通测试使用自身构造的数据。安装 `pytest` 后，在本目录执行：

```powershell
python -m pytest -q tests
```

其中独立宿主内核测试在 Linux 使用 C++17 编译器；可用的 CUDA 测试使用本机 CUDA 环境。
