# 图像识别 + A* + MPC 追逃项目

程序已实现并在 WSL2 Ubuntu-22.04 的 `multiagent-ml` 环境验证。所有扩展程序和输出集中于本目录；未修改老师的物理、追击和终止规则。

当前 `mpc` 使用带避险恢复搜索的 V2，`mpc_v1` 保留旧版。历史测试数据中的 `mpc` 标签代表当时的 V1，不能混同为 V2 成绩。重现旧实验时使用 `--controller-version v1`；新旧配对比较使用 `--methods mpc_v1 mpc`。

## V2 优化结果

| 数据集 | V1 | V2 | 数据用途 |
|---|---:|---:|---|
| 两个原失败案例 | 0/2 | 2/2 | 已参与策略设计，属于开发案例 |
| 原分层 120 场 | 118/120 | 120/120 | 回归测试 |
| 原普通 100 场 | 100/100 | 100/100 | 回归测试 |
| 新独立 60 场 | 59/60 | 60/60 | 新种子、未用来调参的配对测试 |

V2 在预测危险时允许先向临时避险点移动、暂时远离目标，再接回 A* 路线；恢复搜索用 50 步预测。
两个原失败案例分别 45 步和 19 步成功。旧测试中的成功率未下降，但独立测试只多成功一例，样本不足以声称普遍或必然优于旧版。

详见 [优化验证报告](outputs/v2_independent/优化验证报告.md)。恢复搜索增加计算量，不能保证每步满足 0.1 秒的硬实时期限。
一键演示入口自动使用 V2，无需额外安装依赖或训练。

```bash
# 新独立集的新旧配对复现
python wsl-support/vision_mpc/run.py --mode compare --methods mpc_v1 mpc \
  --cases wsl-support/vision_mpc/outputs/v2_independent/cases.json \
  --config wsl-support/vision_mpc/outputs/v2_independent/config.json \
  --output wsl-support/vision_mpc/outputs/v2_reproduction
```

## 快速运行

组员跨平台安装请先读 `wsl-support/team-share/团队交接指南.md`：原生 Windows 使用 `RunWindows.cmd`，Linux/WSL 使用 `run_linux.sh`。下面的 RunVisionMPC.cmd 是原作者本机入口，绑定原作者盘符，不适合直接转发后双击。

Windows 下双击本目录 `RunVisionMPC.cmd`。它会进入 Linux、激活环境，并显示原始环境窗口及带检测框、规划路线、预测轨迹的标注窗口。标注窗口中按 Esc 停止；终端 Ctrl+C 也可停止。

也可以先双击上一级 `EnterLinux.cmd`，随后在项目主目录执行：

```bash
python wsl-support/vision_mpc/run.py --mode demo --show --video --realtime
```

红球为猎手，绿球为被追击者，黑圆是障碍，紫色方块是目标。默认场景是老师 README 的初始位置。原始 `demo.py` 仍然是固定动作演示；运行本程序才使用新控制器。

`--realtime` 将回合节奏限制到约 10 Hz，不代表实时硬截止保证。默认不加该选项则尽快计算。视频按 10 FPS 保存。

## 自定义考核位置

坐标使用世界坐标，x 正方向向右、y 正方向向上，视野范围约 [-1,1]。示例：

```bash
python wsl-support/vision_mpc/run.py --hunter 0.3 0.1 --evader -0.6 0.5 --goal 0.6 -0.6 --show --video --realtime
```

初始化坐标只交给场景创建模块，不直接交给控制器。控制器始终通过 `env.render("rgb_array")[0]` 获取物体位置。初始对象不能重叠；边界附近目标和近距离追击存在失败风险。

## 评估与复现

```bash
# 新的独立随机场景，所有方法采用同一批初始位置
python wsl-support/vision_mpc/run.py --mode compare --count 100 --seed 20261012

# 复现已保存的正式测试；结果放到新目录
python wsl-support/vision_mpc/run.py --mode compare --controller-version v1 \
  --cases wsl-support/vision_mpc/outputs/final_test/cases.json \
  --config wsl-support/vision_mpc/outputs/final_test/config.json \
  --output wsl-support/vision_mpc/outputs/reproduction

# 生成实验报告和静态图表
python wsl-support/vision_mpc/report.py wsl-support/vision_mpc/outputs/reproduction

# 额外消融：预测模型错误地忽略动作延迟，真实环境不变
python wsl-support/vision_mpc/run.py --mode evaluate --method no_delay \
  --cases wsl-support/vision_mpc/outputs/validation/cases.json

# 核心测试
python -m unittest discover -s wsl-support/vision_mpc/tests -v
```

方法：`direct` 直奔目标，`astar` A* 参考路线跟踪，`mpc` 完整方案，`no_hunter` 去掉猎手风险的 MPC，`no_delay` 忽略动作延迟的消融。默认 compare 对比前四种。

注意：`no_hunter` 与 `no_delay` 仍是 V1 的历史消融，并非“V2 去掉某模块”；不能将它们与 V2 的差异完全归因于猎手预测或动作延迟。V2 的直接基线是 `mpc_v1`。

## 已完成的正式测试

### 2026-10-13 新增分层测试

冻结原参数，新增 **10 类、共 120 个位置**，A* 与 MPC 配对运行，共 240 回合：

| 方法 | 成功 | 被捕 | 绿球撞障碍 |
|---|---:|---:|---:|
| A* 跟踪 | 57/120（47.5%） | 40 | 23 |
| 完整 MPC | 118/120（98.3%） | 2 | 0 |

近距离追击 0.08～0.12 组和目标靠障碍组各成功 11/12；其余八组各 12/12。
17 个 MPC 回合记录到猎手接触障碍，均成功；其余 103 回合成功 101 次。
猎手接触来自原环境物理规则，不是被追击者违规，也不能据此证明碰撞与成功的因果关系。

详见 [分层测试报告](outputs/stratified_20261013/分层测试报告.md)，包括置信区间、生成条件和两个失败案例。
这是等额分组的困难场景测试，不代表考核位置的总体分布；不要与下面旧测试直接合并为一个成功率。

```bash
# 复现分层测试（控制参数冻结）
python wsl-support/vision_mpc/run.py --mode compare --methods astar mpc --controller-version v1 \
  --cases wsl-support/vision_mpc/outputs/stratified_20261013/cases.json \
  --config wsl-support/vision_mpc/outputs/stratified_20261013/config.json \
  --output wsl-support/vision_mpc/outputs/stratified_reproduction
python wsl-support/vision_mpc/stratified_report.py wsl-support/vision_mpc/outputs/stratified_reproduction
```

### 原普通随机测试

100 个相同场景、回合上限 500 步、固定控制参数：

| 方法 | 成功 | 被捕 | 撞障碍 |
|---|---:|---:|---:|
| 直奔目标 | 69 | 7 | 24 |
| A* 跟踪 | 72 | 12 | 16 |
| 完整 MPC | 100 | 0 | 0 |
| MPC 去掉猎手预测 | 90 | 10 | 0 |

测试范围：起始坐标 [-0.72,0.72]，对象初始间距至少 0.25，障碍间距至少 0.13。100/100 是此测试集结果，不是任意位置保证，也不代表老师会给满分。主方案成功率的 95% Wilson 区间约为 96.3%～100%。

正式结果位于 `outputs/final_test`，包含 `cases.json`、冻结 `config.json`、逐场 `results.csv`、`summary.json`、图像和轨迹。运行 `report.py` 可生成 `实验报告.md` 和对比图。

额外 8 个压力场景成功 7 次，初始间距 0.12 的近距离追击场景被捕，保存在 `outputs/stress_test`。
在 12 场验证案例中，忽略动作延迟的消融也成功 12 次，因此不能声称这组数据证明延迟补偿提高了成功率；其预测误差需单独比较。数据保存在 `outputs/delay_ablation`。
带标注窗口的完整默认演示成功 74 步，视频在 `outputs/demo_final/mpc_demo.mp4`。

## 模块与实现差异

- `perception.py`：RGB 阈值和连通域；不是 CNN。当前配色固定，直接分割简单且适合 CPU。
- `state_estimator.py`：连续图像差分估计速度。
- `global_planner.py`：检测障碍膨胀，反向距离场与 A*；不硬编码障碍中心作为控制器输入。
- `dynamics.py`：独立动力学和软碰撞模型；固定边框参数来自公开源码。
- `controller.py`、`controller_v1.py`：保留的旧版，分段扰动的反馈轨迹预测、CEM 采样优化；包含一步延迟和到达后的吸收终止逻辑。
- `controller_v2.py`：在旧版预测危险时增加临时避险点、允许暂时远离目标、50 步恢复预测；常规安全回合沿用旧搜索。恢复搜索增加计算量，不提供必胜或硬实时保证。
- `env_adapter.py`：创建和关闭原环境，每回合重新创建以清空动作队列。
- `evaluator.py`：仅在决策后核对终止和视觉误差，允许读取真实状态用于实验指标；这些指标从不输入控制器。
- `visualize.py`、`replay.py`、`report.py`：标注、录制、回放和报告。

与最初设计相比，采用 CEM 反馈采样代替束搜索，直接 RGB 阈值代替 HSV，速度使用单步差分；显式感知失败退出代替持续漏检预测。安全通过大碰撞代价优先选择候选，不提供数学上的碰撞规避保证。A* 无路时使用目标参考并依靠 MPC 碰撞代价，是一个已知局限。

## 常见问题

- `python: can't open file`：确认位于项目主目录，使用上面的完整脚本相对路径。
- 缺少 numpy/cv2：确认提示符前有 `(multiagent-ml)`，不要在 Windows Conda base 中运行。
- localhost 代理提示：当前本地演示不需要网络，通常可忽略。
- 控制器不需要训练，不需要 CUDA 或 GPU。
- 若目标在物理边界之外、初始位置重叠或猎手非常接近，可能无可行解；不要用改动环境规则掩盖失败。

本目录使用的是既有 `multiagent-ml` 环境中的 numpy、scipy、opencv-python、matplotlib 和 pyglet；无需额外安装深度学习训练包。
