import numpy as np
from multiagent import scenarios
from multiagent.environment import MultiAgentEnv


def create(case):
    scenario = scenarios.load('simple_tag.py').Scenario()
    env = MultiAgentEnv(scenario.make_world(), scenario.reset_world, scenario.reward,
                        scenario.observation, done_callback=scenario.is_done, shared_viewer=True)
    env.reset(np.array([case['hunter'], case['evader']], dtype=float), case['goal'],
              [[-.35, .35], [.35, .35], [0., -.35]])
    env._vision_stopped = False
    return env


def close(env):
    for viewer in env.viewers:
        if viewer is not None:
            viewer.close()


def render_rgb(env):
    # Original Viewer clears before switching context. With an annotated second
    # window, explicitly select the original context first to prevent ghost frames.
    if env.viewers[0] is not None:
        env.viewers[0].window.switch_to()
    frame = env.render('rgb_array')[0]
    if not getattr(env, '_vision_close_registered', False):
        def stop_requested():
            env._vision_stopped = True
        env.viewers[0].window.on_close = stop_requested
        env._vision_close_registered = True
    return frame
