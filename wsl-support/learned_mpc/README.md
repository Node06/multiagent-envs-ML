# 图像训练动力学＋MPC

原图像 → 已训练 PixelNet → 图像位置/速度估计 → 三个学习动力学模型 → MPC 搜索有界二维加速度。原教师环境、猎人、碰撞和终止规则不修改。

模型从连续图像运动监督中学习速度保持系数、动作响应和速度上限，并用三层神经网络学习径向接触修正。位置积分保留公开时间步长和运动学结构。这是结构化神经动力学模型，不能称为无物理先验的端到端网络，也不是完整 PETS 复现。

MPC 的普通搜索和恢复搜索都调用学习模型；没有解析动力学兜底。解析 MPC 仅作为单独对照，使用同一训练 CNN。全局 A* 几何规划保留。模型集成均值用于预测；分歧与验证误差可用于风险裕量，系数只在验证集选择。本轮验证选择系数0，因此最终独立测试没有额外的不确定性膨胀，也不能宣称具有安全保证。

## 运行

在项目根目录、Python3.10与现有 PyTorch CPU 环境执行：

```bash
python team_run.py model-mpc --show
python team_run.py tests
```

不加 `--show` 时不显示 OpenCV 窗口；Linux/WSL 使用 EGL 离屏渲染。原生 Windows 的无界面 OpenGL 后端尚未完整验证。

Windows 可使用项目根目录 `Run_LearnedDynamics_Windows.cmd`，或在已激活的 Conda/Python 环境中执行上述命令。文件夹整体移动后仍使用相对位置加载自带的模型，无需原作者的绝对路径。

## 重训练

```bash
python team_run.py model-collect --output wsl-support/learned_mpc/outputs/team_new
python team_run.py model-train --experiment wsl-support/learned_mpc/outputs/team_new --empirical-init --checkpoint world_empirical.pt
python team_run.py model-evaluate --experiment wsl-support/learned_mpc/outputs/team_new --mode validation
python team_run.py model-evaluate --experiment wsl-support/learned_mpc/outputs/team_new --mode test
```

待机或中断后，在同一环境、同一实验目录和原参数后加 `--resume` 可续跑未完成的评估；场景、模型或方法配置不匹配会拒绝续跑。已完成的评估禁止覆盖。独立复测请使用新的实验目录，验证及测试可用 `--seed` 指定新场景种子，测试中的随机障碍种子为该值加1。旧测试种子只适合回归检查，不能用于调参后声称新的独立成绩。

仅复测现成模型时，将 `world_empirical.pt`、`perception.pt` 和 `validation/selection.json` 复制到新的实验目录并保持层级，再运行 `python team_run.py model-evaluate --experiment 新目录 --mode test --seed 2026101603`。不要对现有目录反复运行测试并覆盖旧证据。

监督标签是前后图像中心位移除以时间步长，输入动作是实际待执行动作；没有读取真实位置/速度作为模型输入或监督。训练/验证/测试按完整采集场景分开；危险接触采集与随机障碍采集只是外围重置，不改变物理规则。

本轮目录 `outputs/main_20261014/` 保存数据、三个网络、训练历史和原始闭环 CSV/JSON。`world.pt` 为首轮拟合对照，`world_empirical.pt` 为通过训练集统计初始化后的模型。独立闭环成绩以 `test/summary.json` 为准，不能将单步误差或少量预检成功当作完整成功率。

已完成的524场同场景评估：

| 方法 | 新课程场景 | 随机障碍压力测试 | 默认演示 |
|---|---:|---:|---|
| 训练动力学＋MPC | 100/100 | 30/30 | 成功 |
| 同 CNN 解析 MPC | 100/100 | 30/30 | 成功 |
| 旧 CNN＋PPO Actor | 69/100 | 24/30 | 被捕获 |
| 未训练动力学＋MPC | 97/100 | 28/30 | 成功 |

所有方法使用相同离屏渲染、场景和原始物理规则。100/100的Wilson 95%区间约为96.3%～100%；30/30约为88.6%～100%，不能当作任意新场景必胜保证。新路线没有证明优于解析 MPC；训练模型比未训练版本多成功5场，有限样本不足以证明稳定优越。独立2006条图像运动数据上，单步平均位置预测误差由未训练的1.069像素降至0.389像素，详见 `prediction_ablation.json`。CPU课程场景决策中位数约91毫秒，含 CNN、不含渲染。

`check_rollouts.py` 检查给定记录动作和初始四个视觉邻域的短期开放环误差；真正控制结果以闭环评估为准。已完成测试不可覆盖，后续评估需新实验目录和新场景种子。
