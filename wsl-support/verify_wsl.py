"""Verify real RGB rendering, environment stepping and CPU PPO inference."""
from pathlib import Path
from types import SimpleNamespace
import importlib.metadata as metadata
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
import torch
import scipy
import matplotlib
from multiagent import scenarios
from multiagent.environment import MultiAgentEnv
from agents.PPO.agent import PPO

for name in ('numpy', 'scipy', 'gymnasium', 'pyglet', 'torch', 'opencv-python', 'matplotlib'):
    print(f'{name}: {metadata.version(name)}', flush=True)
scenario = scenarios.load('simple_tag.py').Scenario()
world = scenario.make_world()
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                    scenario.observation, done_callback=scenario.is_done)
try:
    obs = env.reset(np.array([[0., 0.], [.7, .7]]), [-.5, -.5],
                    [[-.35, .35], [.35, .35], [0., -.35]])
    frame = env.render('rgb_array')[0]
    assert frame.shape == (800, 800, 3), frame.shape
    assert frame.dtype == np.uint8 and frame.std() > 0
    for _ in range(10):
        obs, rewards, dones, info = env.step(np.array([0., 1., 0., 0., 1.]), 1)
        assert np.isfinite(np.asarray(obs)).all()
        if any(dones):
            break
    output = Path(__file__).resolve().parent / 'wsl_render_check.png'
    assert cv2.imwrite(str(output), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cfg = SimpleNamespace(device=torch.device('cpu'), lr=1e-4, gamma=.99,
                          eps_clip=.2, K_epochs=1, train=False)
    agent = PPO(120, 2, cfg)
    action = agent.select_action(np.zeros(120, dtype=np.float32))
    assert np.isfinite(np.asarray(action)).all()
    print(f'PASS: RGB={frame.shape}, steps=10, PPO CPU action={action}', flush=True)
    print(f'Preview: {output.resolve()}', flush=True)
finally:
    for viewer in env.viewers:
        if viewer is not None:
            viewer.close()
