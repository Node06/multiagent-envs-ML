import numpy as np
from vision_mpc.env_adapter import create, close, render_rgb
from vision_mpc.perception import detect, DetectionError
from vision_mpc.state_estimator import Estimator
from vision_mpc.geometry import action5, limit
from vision_mpc.evaluator import terminal_reason
from .observation import encode
from .rewards import potential, reward


class VisualEnv:
    def __init__(self, cfg):
        self.cfg, self.env = cfg, None

    def reset(self, case):
        if self.env is None:
            self.env = create(case)
        else:
            # Same original reset; keep the GL window/context for training speed.
            self.env.reset(np.array([case['hunter'], case['evader']]), case['goal'],
                           [[-.35, .35], [.35, .35], [0., -.35]])
        # Original reset does not clear queued controls. Do so before any physics.
        for agent in self.env.world.agents:
            agent.action.u = np.zeros(2)
            agent.action.c = np.zeros(2)
        self.env.world.buffer.clear()
        self.env._vision_stopped = False
        self.pending = np.zeros(2)
        self.estimator = Estimator(self.cfg['dt'])
        self.steps = 0
        self.state = self.estimator.update(detect(render_rgb(self.env)))
        for viewer in self.env.viewers:
            viewer.window.set_vsync(False)
            viewer.window.set_visible(False)
        self.phi = potential(self.state, self.cfg)
        return encode(self.state, self.pending)

    def step(self, action):
        # Raw tanh action is what PPO models. Projection is a deterministic
        # environment transform and agrees with MPC's action5.
        self.pending = limit(np.asarray(action, dtype=float), 1.)
        _, _, done, _ = self.env.step(action5(self.pending))
        self.steps += 1
        reason = terminal_reason(self.env, done)  # adjudication only
        try:
            self.state = self.estimator.update(detect(render_rgb(self.env)))
            next_phi = potential(self.state, self.cfg)
        except DetectionError:
            # No oracle fallback. The retained observation is terminal-only.
            next_phi = self.phi
            reason = reason or 'perception_failure'
        if getattr(self.env, '_vision_stopped', False):
            raise KeyboardInterrupt('Environment window closed')
        terminated = reason is not None
        truncated = not terminated and self.steps >= self.cfg['max_steps']
        if truncated:
            reason = 'timeout'
        r = reward(self.phi, next_phi, reason, self.cfg)
        self.phi = next_phi
        return encode(self.state, self.pending), r, terminated, truncated, {'reason': reason}

    def close(self):
        if self.env is not None:
            close(self.env)
            self.env = None
