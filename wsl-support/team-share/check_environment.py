"""Cross-platform dependency, RGB rendering and image-controller smoke check."""
import argparse
import importlib.metadata as metadata
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'wsl-support'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--packages-only', action='store_true')
    p.add_argument('--ppo', action='store_true', help='Also check CPU-loadable PPO model and 21-feature inference')
    args = p.parse_args()
    print('OS:', platform.system(), 'Python:', sys.version.split()[0], 'Executable:', sys.executable)
    if sys.version_info[:2] != (3, 10):
        raise SystemExit('Use Python 3.10 for the shared baseline (legacy code imports imp).')
    expected = {'numpy':'1.26.4', 'scipy':'1.14.1', 'gymnasium':'0.29.1', 'pyglet':'1.5.24',
                'six':'1.17.0', 'opencv-python':'4.10.0.84', 'matplotlib':'3.9.4'}
    for name, version in expected.items():
        installed = metadata.version(name)
        print(f'{name}: {installed}')
        if installed != version:
            raise SystemExit(f'{name}: expected {version}, reinstall requirements-team.txt')
    if args.ppo:
        installed = metadata.version('torch')
        print('torch:', installed)
        if installed.split('+')[0] != '2.6.0':
            raise SystemExit('Use torch 2.6.0 for the shared PPO baseline')
    if args.packages_only:
        print('PASS: package baseline. Rendering has NOT been checked.')
        return
    import json
    import numpy as np
    import cv2
    from vision_mpc.env_adapter import create, render_rgb, close
    from vision_mpc.perception import detect
    from vision_mpc.state_estimator import Estimator
    from vision_mpc.global_planner import Planner
    from vision_mpc.controller_v2 import Controller
    from vision_mpc.geometry import action5
    case = {'id':'platform_check', 'hunter':[0.,0.], 'evader':[.7,.7], 'goal':[-.5,-.5]}
    env = create(case)
    try:
        image = render_rgb(env)
        assert image.shape == (800,800,3), f'Unexpected framebuffer size: {image.shape}; check display scaling'
        assert image.dtype == np.uint8
        state = Estimator(.1).update(detect(image))
        cfg = json.loads((ROOT/'wsl-support'/'vision_mpc'/'config.json').read_text())
        controller = Controller(cfg, Planner(state.detection,cfg))
        q, _ = controller.choose(state)
        env.step(action5(q))
        target = ROOT/'wsl-support'/'team-share'/'platform_check.png'
        assert cv2.imwrite(str(target),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        print('PASS: RGB rendering, all-object detection and V2 action. Preview:',target)
    finally:
        close(env)
    if args.ppo:
        import torch
        from vision_ppo.env_wrapper import VisualEnv
        from vision_ppo.model import ActorCritic
        from vision_ppo.checkpoint import resolve_checkpoint
        folder = ROOT/'wsl-support/vision_ppo/outputs/formal_20261013'
        selection = json.loads((folder/'selected_model.json').read_text())
        checkpoint = resolve_checkpoint(selection, folder)
        cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
        torch.set_num_threads(1)
        model = ActorCritic(cp['config']['hidden']); model.load_state_dict(cp['model']); model.eval()
        visual = VisualEnv(cp['config'])
        try:
            obs = visual.reset(case)
            assert obs.shape == (21,) and np.isfinite(obs).all()
            with torch.no_grad():
                action = model.deterministic(torch.from_numpy(obs)).numpy()
            assert action.shape == (2,) and np.isfinite(action).all()
            next_obs, _, _, _, _ = visual.step(action)
            assert np.isfinite(next_obs).all()
            print('PASS: portable checkpoint, CPU PPO inference and original RGB environment step.', checkpoint)
        finally:
            visual.close()


if __name__ == '__main__':
    main()
