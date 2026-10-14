# Linux 项目环境

环境已在 WSL2 Ubuntu-22.04 配置。Python 3.10.12，虚拟环境路径为
`/home/lin_node/.venvs/multiagent-ml`。使用 CPU 版 PyTorch 2.6.0。
Windows 的其他 Python 和 d2l 环境未修改。

## 进入环境

在 Windows 中双击本目录的 `EnterLinux.cmd`，会进入 Linux、激活虚拟环境，
并切换到当前项目目录。终端提示符前的 `(multiagent-ml)` 表示环境已激活。

也可在 PowerShell 输入：

```powershell
wsl -d Ubuntu-22.04
```

随后在 Linux 输入：

```bash
source ~/.venvs/multiagent-ml/bin/activate
cd /mnt/e/Materials/MachineLearning/lab/multiagent-envs-ML
python demo.py
```

## 验证与运行

```bash
python wsl-support/verify_wsl.py
python demo.py
```

`verify_wsl.py` 检查依赖、真实 800×800 RGB 渲染、环境推进和 PPO CPU 推理，
并在 `wsl-support` 文件夹中生成 `wsl_render_check.png`。配置时这些检查全部通过，`pip check` 无依赖冲突。
尚未进行完整 PPO 训练。

本次新增的配置、验证脚本及图片均集中在 `wsl-support` 文件夹。
完整依赖版本保存在该文件夹的 `requirements-wsl.lock.txt`。
`configure_wsl.sh` 包含本次 Python 环境的配置命令；重建环境前需要系统已有
`python3-venv libglu1-mesa libgl1 libglib2.0-0`。
PyTorch CPU 包来自官方 https://download.pytorch.org/whl/cpu 。

项目 `setup.py` 的旧 `gym` / `numpy-stl` 声明已改为实际运行所需的
Gymnasium、NumPy、Pyglet 和 six；没有修改算法代码。

退出虚拟环境输入 `deactivate`，退出 Linux 终端输入 `exit`。

## 完成的视觉规划项目

新控制器位于 `wsl-support/vision_mpc`，使用图像识别、A* 与 MPC 自动避障避追。
使用方法及正式实验见该目录 `README.md`。双击 `vision_mpc/RunVisionMPC.cmd` 可一键演示。

在已经激活环境的项目主目录中也可运行：

```bash
python wsl-support/vision_mpc/run.py --mode demo --show --video --realtime
```

它与原始固定动作 `demo.py` 是不同入口。
