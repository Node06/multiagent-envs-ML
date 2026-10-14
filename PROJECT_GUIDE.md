# Save the Prey：学习模型主方案

当前提交主路线是 **RGB 图像 → 训练 CNN 感知 → 训练 Actor → 二维有界加速度**。控制网络通过 MPC 示范、DAgger 和 PPO 微调训练；MPC 只用于训练标签和对照，主模型运行时不调用 MPC。对外接口 `LearnedController.act(rgb)` 不接受真实状态。

教师原环境 `multiagent/`、参考 `agents/` 与原根目录代码保持基线 ca0f9ab，不修改猎人行为、速度、物理碰撞或终止规则。跨平台完整性检查只标准化 Git 的 CRLF/LF 换行差异。

## 先读哪些资料

1. [课设要求核验](wsl-support/course_audit/课设要求核验.md)：PPT第5～8页、固定障碍要求、图像输入、机器学习要求和猎人行为。
2. [当前学习模型说明](wsl-support/learned_control/README.md)：模型训练、验证、PPO选模、独立测试和限制。
3. [团队跨平台安装](wsl-support/team-share/团队交接指南.md)：Python3.10、Windows/Conda/WSL依赖。

原根目录 readme.md 保留作为教师材料，其中“或者其他方法”不能替代新 PPT 的明确机器学习要求。原 `vision_ppo/` 是旧的视觉状态纯 PPO 基线；`vision_mpc/` 是规划与示范基线。新的提交模型在 `learned_control/`。

已完成 450 场配对独立评估：课程固定障碍主模型 42/60（70%），最佳模仿模型 39/60，旧纯 PPO 13/60，MPC 60/60；随机障碍压力测试主模型 22/30。默认演示场景主模型在第22步被捕获，不能当作成功演示。PPO 相对模仿模型仅多成功3场，单种子实验不能证明算法优越。完整结果、失败分布和定位误差见 `wsl-support/learned_control/outputs/main_20261014/主模型实验报告.md`。

## Windows 与 Linux 共用命令

在 Python3.10 环境、项目根目录执行；不需要 NVIDIA 或 CUDA：

```bash
python team_run.py check --ppo
python wsl-support/course_audit/audit.py --verify-only
python team_run.py learned --show
python team_run.py mpc --show --video --realtime
```

Windows 可从已激活的 Conda 终端运行 `Run_Learned_Windows.cmd`；使用项目 `.venv` 时可直接双击该文件。旧 Run_PPO_Windows.cmd 展示的是旧纯 PPO，保留为基线。

## 继续训练与评估

```bash
python team_run.py learned-train --seed 2401 --output wsl-support/learned_control/outputs/team2401
python team_run.py learned-evaluate --experiment wsl-support/learned_control/outputs/team2401 --seed 2026101504 --output wsl-support/learned_control/outputs/team2401/new_test
python team_run.py learned --checkpoint wsl-support/learned_control/outputs/team2401/primary_ppo.pt --show
```

固定障碍是课程主分布；两球与目标在合法范围随机。随机三障碍布局是单独的额外压力测试。评估完成后才根据验证日志导出 `primary_ppo.pt`，只在 PPO 微调阶段的检查点之间选择；独立测试不反馈选模。

RGB 输入由训练 CNN 提取语义掩码，然后内部计算视觉位置与速度。这是分阶段训练的结构化图像管线，不能宣称整条链条可微端到端。禁止把 `env.world` 或 `env.observation()` 的坐标作为模型输入。

## Git 与实验记录

- 新分支修改 `learned_control/`，保留原环境和已发布基线。
- 验证集可调参；已公开测试集只能用于回归，最终性能使用新种子/新场景。
- 发布时一起保留模型、训练配置、场景、逐场 CSV、总结、失败视频和原代码完整性记录。
- 大规模训练缓存、所有临时视频与重复打包文件继续忽略；最终发布模型和精简结果由维护者有选择地提交。
