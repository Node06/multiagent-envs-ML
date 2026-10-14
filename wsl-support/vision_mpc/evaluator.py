import numpy as np


def hunter_obstacle_distance(env):
    """Post-action diagnostics only; contact is not a terminal condition."""
    hunter = env.world.agents[0]
    return float(min(np.linalg.norm(hunter.state.p_pos-obstacle.state.p_pos)
                     for obstacle in env.world.landmarks))


def perception_error_pixels(env, detection):
    """Diagnostics only, called after command selection, never fed back."""
    return float(max(np.linalg.norm(detection.hunter-env.world.agents[0].state.p_pos),
                     np.linalg.norm(detection.evader-env.world.agents[1].state.p_pos))*400)


def wilson(successes, count):
    z = 1.96
    p = successes/count
    center = (p+z*z/(2*count))/(1+z*z/count)
    half = z*np.sqrt(p*(1-p)/count+z*z/(4*count*count))/(1+z*z/count)
    return [float(center-half), float(center+half)]


def terminal_reason(env, done):
    """Metrics-only adjudicator; true coordinates never enter the controller."""
    if not any(done):
        return None
    hunter, evader = env.world.agents
    p = evader.state.p_pos
    if np.linalg.norm(p-hunter.state.p_pos) < .05:
        return 'caught'
    if any(np.linalg.norm(p-o.state.p_pos) <= .075 for o in env.world.landmarks):
        return 'obstacle_collision'
    if np.linalg.norm(p-env.world.check[0].state.p_pos) < .075:
        return 'success'
    return 'ambiguous_terminal'


def generate_cases(count, seed):
    rng = np.random.default_rng(seed)
    obstacles = np.array([[-.35, .35], [.35, .35], [0., -.35]])
    cases = []
    while len(cases) < count:
        points = rng.uniform(-.72, .72, size=(3, 2))
        if np.min(np.linalg.norm(points[:, None]-obstacles[None], axis=-1)) < .13:
            continue
        if min(np.linalg.norm(points[i]-points[j]) for i in range(3) for j in range(i)) < .25:
            continue
        cases.append({'id': f's{seed}_{len(cases):03}', 'hunter': points[0].tolist(),
                      'evader': points[1].tolist(), 'goal': points[2].tolist()})
    return cases
