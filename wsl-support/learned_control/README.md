# 图像输入的示范学习与 PPO 主模型

教师 PPT 要求至少一种机器学习、训练能够避障避猎人并到达目标的模型，输入图像、输出二维有界加速度。本目录实现训练 CNN 感知、示范预训练、DAgger、PPO 微调，以及独立配对测试。

## 算法与训练

- PixelNet：三层 1×1 卷积学习 RGB 语义类别，监督标签来自 RGB 分割；不使用真实坐标标签。用400×400图像推理，连通域与像素尺度转换得到内部视觉坐标。
- Actor/Critic：两层256单元，控制网络接收内部21维视觉特征；公开 `LearnedController.act(rgb)` 接收原RGB图像并维护图像速度估计和自身待执行动作。
- MPC V2：只生成专家动作，全部来自图像估计状态。160个随机初始场景示范，随后3轮DAgger，每轮60个新场景；执行学生动作时同步专家的待执行动作，避免标签预测了错误动作队列。
- PPO：从验证集最佳模仿模型开始，学习率3e-5、低探索标准差、16384步微调，附加示范动作正则以减少遗忘。每4096步进行固定40场验证。
- 主模型 `primary_ppo.pt` 只在 PPO 阶段按验证成功率、成功步数选出；`imitation_best.pt` 保留预训练/DAgger最佳模型，检验 PPO 收益。`best.pt` 是所有阶段的验证最佳历史快照，可能不是 PPO 模型，不能混淆。

这不是可微端到端视觉控制网络：图像分类和控制分阶段训练，内部有连通域及显式视觉坐标。它的外部输入不包括环境真实状态；README与PPT的区别及评分解释见 course_audit。不会保证教师给分，也不会称 PPO 已经被证明优于 SAC。

## 运行

安装依赖同 team-share，使用 Python3.10 与 PyTorch2.6.0；Windows不必安装Linux。

```bash
python team_run.py learned --show
python team_run.py learned-train --output wsl-support/learned_control/outputs/new_training
python team_run.py learned-evaluate --experiment wsl-support/learned_control/outputs/new_training --seed 2026101504
python wsl-support/course_audit/audit.py --verify-only
```

原始代码不修改；场景重置、感知、训练和统计在外围模块完成。主模型运行时没有 A* 或 MPC 兜底；检测失败明确计入失败。

## 实验用途

- 40个固定验证场景：选阶段与模型。
- 新的60场固定障碍随机两球/目标：课程分布独立测试。
- 新的30场随机三障碍：分布外压力测试，单独报告。
- 同批场景比较 A*、MPC、旧纯PPO、最佳模仿模型、PPO主模型。
- 逐帧统计 CNN 的定位误差、猎人排队加速度对齐、猎人接触障碍帧数；真实坐标仅做事后诊断。

训练日志、模型、验证记录和最终评估位于 `outputs/main_20261014/`。实验运行中不将验证结果写成最终测试成绩。后续结果汇总位于 `outputs/main_20261014/主模型实验报告.md`。
