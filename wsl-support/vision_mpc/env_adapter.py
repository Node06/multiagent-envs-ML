import numpy as np
import sys
import pyglet
from threading import RLock
if sys.platform.startswith('linux'):
    # EGL pbuffer rendering, not an unmapped X11 window (which can read blank).
    pyglet.options['headless'] = True
from multiagent import scenarios
from multiagent.environment import MultiAgentEnv

_RENDER_LOCK = RLock()


class _HiddenWindowModule:
    def __init__(self, module):
        self._module = module

    def Window(self, *args, **kwargs):
        kwargs['visible'] = False
        window = self._module.Window(*args, **kwargs)
        if sys.platform == 'win32':
            # Hidden Win32 windows may never receive the initial resize event.
            # Initialize pyglet's pixel projection before the first RGB render.
            window.switch_to()
            window.on_resize(window.width, window.height)
        return window

    def __getattr__(self, name):
        return getattr(self._module, name)


class _HiddenPyglet:
    """Local rendering facade; the real pyglet Window class stays unchanged."""
    def __init__(self, module):
        self._module = module
        self.window = _HiddenWindowModule(module.window)

    def __getattr__(self, name):
        return getattr(self._module, name)


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
    # The teacher's Viewer creates a visible window even for rgb_array.
    # Construct only its RGB context with visible=False, before it can map or
    # steal focus. Explicit --show annotation windows are created separately.
    from multiagent import rendering
    with _RENDER_LOCK:
        real_pyglet = rendering.pyglet
        rendering.pyglet = _HiddenPyglet(real_pyglet)
        try:
            frame = env.render('rgb_array')[0]
        finally:
            rendering.pyglet = real_pyglet
    if not getattr(env, '_vision_close_registered', False):
        def stop_requested():
            env._vision_stopped = True
        env.viewers[0].window.on_close = stop_requested
        env._vision_close_registered = True
    return frame
