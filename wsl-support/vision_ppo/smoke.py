"""Real-render interface checks and throughput measurement."""
import json
import sys
import time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent)]
from vision_ppo.env_wrapper import VisualEnv
from vision_mpc.evaluator import generate_cases

cfg = json.loads((HERE/'config.json').read_text()); cfg['max_steps'] = 30
case = generate_cases(1, 991)[0]
env = VisualEnv(cfg)
try:
    obs = env.reset(case)
    assert obs.shape == (21,) and np.isfinite(obs).all()
    first = env.state.detection.evader.copy()
    obs, _, terminal, truncated, _ = env.step([.1, 0.])
    np.testing.assert_allclose(env.state.detection.evader, first)  # queued action delay
    env.step([.1, 0.])
    assert env.state.detection.evader[0] > first[0]
    env.reset(case)
    np.testing.assert_allclose(env.pending, 0)
    np.testing.assert_allclose(env.state.evader_velocity, 0)
    for agent in env.env.world.agents:
        np.testing.assert_allclose(agent.action.u, 0)
    start = time.perf_counter()
    for _ in range(300):
        obs, _, terminal, truncated, info = env.step([0., 0.])
        assert np.isfinite(obs).all()
        if terminal or truncated:
            env.reset(case)
    print(json.dumps({'real_render_checks': 'passed', 'steps': 300,
                      'seconds': time.perf_counter()-start}), flush=True)
finally:
    env.close()
