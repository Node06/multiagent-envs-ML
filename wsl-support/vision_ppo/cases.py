import numpy as np
from vision_mpc.evaluator import generate_cases


OBSTACLES = np.array([[-.35, .35], [.35, .35], [0., -.35]])


def training_case(rng, progress, fraction):
    # Original pursuer remains active at every stage; vary starts only.
    stage = min(1., progress/max(fraction, 1e-9))
    for _ in range(10000):
        evader = rng.uniform(-.72, .72, 2)
        hunter = rng.uniform(-.72, .72, 2)
        if rng.random() < .8*(1-stage):
            angle = rng.uniform(-np.pi, np.pi)
            distance = rng.uniform(.15, .3+.7*stage)
            goal = evader + distance*np.array([np.cos(angle), np.sin(angle)])
        else:
            goal = rng.uniform(-.72, .72, 2)
        points = np.array([hunter, evader, goal])
        if np.max(np.abs(points)) > .75:
            continue
        if np.min(np.linalg.norm(points[:, None]-OBSTACLES[None], axis=-1)) < .13:
            continue
        if np.linalg.norm(evader-goal) < .12 or np.linalg.norm(hunter-goal) < .15:
            continue
        if np.linalg.norm(hunter-evader) < .25+.55*(1-stage):
            continue
        return {'id': 'training', 'hunter': hunter.tolist(), 'evader': evader.tolist(), 'goal': goal.tolist()}
    raise RuntimeError('Could not sample a valid curriculum case')
