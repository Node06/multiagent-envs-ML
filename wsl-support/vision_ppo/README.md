# 视觉状态 PPO 对照方案

**团队收件人请先读 [跨平台交接指南](../team-share/团队交接指南.md)。Windows 不必使用 Linux；解压后可运行 `python team_run.py check --ppo` 和 `python team_run.py ppo --show`。下文的 `/mnt/e/...`、旧 RunPPO.cmd 和既有 venv 路径属于作者的历史复现命令，共享包采用新的相对路径入口。**

独立控制绿球的深度强化学习策略，与原版 A*＋MPC 做配对实验。采用原环境 800×800 RGB 图像，经共享颜色分割和连续帧差分得到 21 维状态，策略网络没有真实坐标输入。红球维持原追击规则。

## 已完成的正式实验

2026-10-13：三个种子各训练 131072 步，共 393216 步；使用验证集选出的冻结检查点，在 60 个新场景上与 A*、原版 MPC 配对测试，共 300 回合。

| 方法 | 成功 | 成功率 |
|---|---:|---:|
| A* 跟踪 | 36/60 | 60.0% |
| 原版 MPC | 60/60 | 100.0% |
| PPO seed 11 | 12/60 | 20.0% |
| PPO seed 23 | 7/60 | 11.7% |
| PPO seed 37 | 3/60 | 5.0% |

演示模型依据验证集选择 seed 11 在 65536 步的 best.pt（验证 45%，独立测试 20%），不是依据测试集挑选。其控制计算约 0.28 ms，含物理推进和视觉的单步约 14.17 ms；MPC 分别约 49.43 ms、64.21 ms。以上时间为各回合中位数的均值，不含初始化。

本轮固定训练预算下 PPO 成功率低于 MPC，未证明其能替代主控制器。主要失败为撞障碍和被捕；训练完成不等于已收敛。详见[实验报告](outputs/formal_20261013/实验报告.md)、[逐场结果](outputs/formal_20261013/holdout/results.csv)和[模型选择记录](outputs/formal_20261013/selected_model.json)。

成功、被捕、撞障碍三类回放均已录制，并核对终止原因及步数与冻结测试一致。可查看[成功回放](outputs/formal_20261013/representative_replays/ppo_seed11_s6101399_001.mp4)及[回放缩略图](outputs/formal_20261013/representative_replays/replay_contact_sheet.png)。默认初始场景下该 PPO 模型在第 171 步撞障碍，RunPPO.cmd 默认展示这一场景，属于失败演示；没有用成功案例替换默认场景。

## 环境

使用已经配置的 WSL Ubuntu-22.04 `~/.venvs/multiagent-ml`。需要 CPU PyTorch、numpy、opencv-python、gymnasium、pyglet、matplotlib 和可用 OpenGL 显示。已检查 PyTorch 2.6.0+cpu；无需 CUDA。所有命令从项目根目录执行。

```bash
source ~/.venvs/multiagent-ml/bin/activate
cd /mnt/e/Materials/MachineLearning/lab/multiagent-envs-ML
```

## 验证与训练

```bash
python -m unittest discover -s wsl-support/vision_ppo/tests -v
python wsl-support/vision_ppo/smoke.py

# 短程检查（独立输出，不作为正式实验）
python wsl-support/vision_ppo/train.py --seed 11 --steps 8192 \
  --output wsl-support/vision_ppo/outputs/short_run

# 完整配对实验：三个独立训练进程，每个 131072 步，冻结后测试
python wsl-support/vision_ppo/run_experiment.py \
  --output wsl-support/vision_ppo/outputs/reproduction --steps 131072 --test-count 60

# 单种子默认配置预算为 262144 步
python wsl-support/vision_ppo/train.py --seed 11 \
  --output wsl-support/vision_ppo/outputs/long_seed11
```

CPU 图像采样占主要训练时间。训练窗口隐藏；通过每个 seed 的日志查看进展。按 Ctrl+C 中止时，保留最近完整更新的 latest.pt，不把半条 rollout 当成可恢复检查点。

## 恢复训练

```bash
python wsl-support/vision_ppo/train.py --seed 11 --steps 262144 \
  --resume wsl-support/vision_ppo/outputs/long_seed11/latest.pt \
  --output wsl-support/vision_ppo/outputs/long_seed11
```

恢复网络、优化器和随机数状态，重新初始化回合；不保证与不中断运行逐位相同。更改总预算会改变课程进度，因此应在独立实验中记录。其余配置必须与检查点一致。保存 best.pt、latest.pt 和 final.pt，最佳模型只由验证场景选择。

## 独立评估与演示

```bash
python wsl-support/vision_ppo/evaluate.py \
  --checkpoints wsl-support/vision_ppo/outputs/long_seed11/best.pt \
  --cases wsl-support/vision_ppo/outputs/formal_20261013/holdout/cases.json \
  --baselines astar mpc_v1 --output wsl-support/vision_ppo/outputs/retest

python wsl-support/vision_ppo/demo.py --show
python wsl-support/vision_ppo/demo.py --show \
  --checkpoint wsl-support/vision_ppo/outputs/long_seed11/best.pt \
  --hunter 0.3 0.1 --evader -0.6 0.5 --goal 0.6 -0.6
```

`demo.py` 默认使用正式实验按验证集选出的模型，生成 MP4 和轨迹。若尚未完成正式实验，请显式指定检查点。也可以双击 RunPPO.cmd。

完成实验后，可以生成各类典型结果的视频；回放会核对结果及步数与冻结测试一致：

```bash
python wsl-support/vision_ppo/media.py wsl-support/vision_ppo/outputs/formal_20261013
```

## 实现细节

- observation.py：21 维固定尺度状态，包含位置、图像估计速度、目标/障碍相对位置、半径和待执行动作；不做整向量单位长度归一化。
- env_wrapper.py：统一 RGB 输入、动作投影、原一步延迟、回合重置和终止分类。重置清空旧动作；训练时复用 GL 上下文。
- model.py：独立 Actor/Critic，两层 128 单元 MLP；可学习标准差和 tanh 有界高斯策略。
- buffer.py / ppo.py：GAE、超时自举、跨回合截断、mini-batch clipped PPO、熵正则、梯度裁剪和 KL 提前停止。
- rewards.py：成功/失败事件、时间惩罚及势函数目标距离塑形。终止状态势函数为零，超时状态保留价值自举。
- train.py：随机场景课程学习、20 场固定验证、模型保存与恢复。
- evaluate.py：冻结模型与 A*、MPC 的相同场景测试，保存逐场结果、模型校验值和可选视频。
- run_experiment.py / report.py：三个种子完整实验、学习曲线、成功率区间、配对差异及中文报告。

本次对共享 perception.py 的修改仅将原 RGB 阈值运算换成等价 OpenCV 内核以节省 CPU 分配；阈值、连通域筛选和坐标换算保持一致，随机像素等价测试及原 MPC 测试覆盖该修改。

## 公平性与限制

真实坐标仅用于初始化、原物理推进和结果判定，不进入策略或奖励距离塑形。训练奖励与原示例不同，但最终比较遵循原成功/失败规则。验证集选模型；新测试场景在所有模型冻结后生成，不反馈训练。

默认对照 mpc_v1 明确对应原版控制器；可指定 mpc_v2 比较当前改进版。测试成功率是指定分布的结果，不保证任意初始位置可解。固定预算训练完成不代表 PPO 已收敛。失败回合不可计为成功，也不可只展示成功视频来推断整体表现。
