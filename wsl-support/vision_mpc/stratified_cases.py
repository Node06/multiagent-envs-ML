"""Generate predeclared challenge strata; reject invalid starts, never failures."""
import argparse
import json
from pathlib import Path
import numpy as np

GROUPS = {
    'ordinary': '普通随机位置（沿用旧范围）',
    'hunter_gap_008_012': '近距离追击：0.08～0.12',
    'hunter_gap_012_020': '近距离追击：0.12～0.20',
    'hunter_gap_020_035': '近距离追击：0.20～0.35',
    'hunter_blocks_route': '猎手位于起点到目标连线附近',
    'edge_goal': '目标靠边：单轴绝对值0.72～0.80',
    'near_obstacle_start': '绿球靠障碍：中心距离0.082～0.11',
    'near_obstacle_goal': '目标靠障碍：中心距离0.08～0.12',
    'corner_start': '绿球靠角：双轴绝对值0.73～0.80',
    'long_route': '远距离目标：起终点距离至少1.35'
}
OBSTACLES = np.array([[-.35, .35], [.35, .35], [0., -.35]])


def valid(points, group):
    h, e, g = points
    if not np.isfinite(points).all() or np.max(abs(points)) > .800001:
        return False
    distances = np.linalg.norm(points[:, None]-OBSTACLES[None], axis=-1)
    if distances[0].min() <= .085 or distances[1].min() <= .080 or distances[2].min() <= .078:
        return False
    if np.linalg.norm(h-e) <= .075 or min(np.linalg.norm(g-h), np.linalg.norm(g-e)) <= .15:
        return False
    if group == 'ordinary':
        return distances.min() >= .13 and min(np.linalg.norm(h-e), np.linalg.norm(h-g), np.linalg.norm(e-g)) >= .25
    if group == 'long_route':
        return np.linalg.norm(e-g) >= 1.35 and np.linalg.norm(h-e) >= .3
    return True


def generate(per_group=12, seed=20261013):
    rng = np.random.default_rng(seed)
    cases = []
    for group in GROUPS:
        accepted = 0
        for _ in range(100000):
            points = rng.uniform(-.72, .72, size=(3, 2))
            h, e, g = points
            if group.startswith('hunter_gap_'):
                low, high = {'hunter_gap_008_012': (.08, .12), 'hunter_gap_012_020': (.12, .20),
                             'hunter_gap_020_035': (.20, .35)}[group]
                angle = rng.uniform(0, 2*np.pi)
                h[:] = e+rng.uniform(low, high)*np.array([np.cos(angle), np.sin(angle)])
            elif group == 'hunter_blocks_route':
                if np.linalg.norm(e-g) < .8:
                    continue
                delta = g-e
                perpendicular = np.array([-delta[1], delta[0]])/np.linalg.norm(delta)
                h[:] = e+rng.uniform(.3, .7)*delta+rng.uniform(-.035, .035)*perpendicular
            elif group == 'edge_goal':
                axis = rng.integers(2)
                g[axis] = rng.choice([-1, 1])*rng.uniform(.72, .8)
            elif group.startswith('near_obstacle_'):
                point = e if group == 'near_obstacle_start' else g
                center = OBSTACLES[rng.integers(3)]
                angle = rng.uniform(0, 2*np.pi)
                bounds = (.082, .11) if group == 'near_obstacle_start' else (.08, .12)
                point[:] = center+rng.uniform(*bounds)*np.array([np.cos(angle), np.sin(angle)])
            elif group == 'corner_start':
                e[:] = rng.choice([-1, 1], size=2)*rng.uniform(.73, .8, size=2)
            if not valid(points, group):
                continue
            cases.append({'id': f'{group}_{accepted:03}', 'group': group,
                          'hunter': h.tolist(), 'evader': e.tolist(), 'goal': g.tolist(),
                          'initial_hunter_gap': float(np.linalg.norm(h-e)),
                          'initial_evader_obstacle_distance': float(np.linalg.norm(e-OBSTACLES, axis=1).min())})
            accepted += 1
            if accepted == per_group:
                break
        if accepted != per_group:
            raise RuntimeError(f'Could not populate group {group}')
    return cases


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--per-group', type=int, default=12)
    p.add_argument('--seed', type=int, default=20261013)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.per_group < 1:
        p.error('--per-group must be positive')
    cases = generate(args.per_group, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'{len(cases)} cases -> {args.output}')
