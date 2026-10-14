@echo off
wsl -d Ubuntu-22.04 -- bash -lc "source ~/.venvs/multiagent-ml/bin/activate; cd /mnt/e/Materials/MachineLearning/lab/multiagent-envs-ML; python wsl-support/vision_ppo/demo.py --show"
pause
