import numpy as np
from .geometry import limit, unit, segment_distance
from .dynamics import advance, borders


class Controller:
    """Feedback shooting MPC with CEM; real environment is never used for rollouts."""
    def __init__(self, config, planner, method='mpc'):
        self.cfg, self.planner, self.method = config, planner, method
        self.rng = np.random.default_rng(config['seed'])
        self.pending = np.zeros(2)
        self.started = False
        self.route = None
        self.counter = 0
        self.mean = np.zeros((int(np.ceil(config['horizon']/config['chunk_steps'])), 2))

    def choose(self, state):
        d, cfg = state.detection, self.cfg
        if self.route is None or self.counter % 10 == 0:
            self.route = self.planner.plan(d.evader)
        self.counter += 1
        target = self.planner.waypoint(d.evader[None], self.route, cfg['lookahead'])[0]
        if self.method == 'direct':
            target = d.goal
        nominal = limit((unit(target-d.evader)*.25-state.evader_velocity)*8, 1.)
        if self.method in ('direct', 'astar'):
            self.pending, self.started = nominal, True
            return nominal, {'route': self.route, 'predicted': [], 'unsafe': False}
        samples = cfg['samples']
        mean, std = self.mean.copy(), np.full_like(self.mean, .65)
        best = None
        for _ in range(cfg['iterations']):
            noise = self.rng.normal(mean, std, size=(samples, *mean.shape))
            noise[0] = 0
            noise[1] = mean
            # Explicit turning proposals avoid a purely local goal-following search.
            noise[2:6, :, :] = np.array([[0, .8], [0, -.8], [.8, 0], [-.8, 0]])[:, None, :]
            result = self.rollout(state, noise)
            order = np.argsort(result[0])
            elite = noise[order[:max(6, samples//8)]]
            mean = elite.mean(axis=0)
            std = np.maximum(elite.std(axis=0), .15)
            pick = order[0]
            if best is None or result[0][pick] < best[0]:
                best = (result[0][pick], result[1][pick], result[2][:, pick], noise[pick], result[3][pick])
        self.mean = best[3]*.8
        command = best[1]
        self.pending, self.started = command.copy(), True
        return command, {'route': self.route, 'predicted': best[2], 'unsafe': bool(best[4]), 'cost': float(best[0])}

    def rollout(self, state, perturbations):
        d, cfg = state.detection, self.cfg
        n = len(perturbations)
        p = np.tile(d.evader, (n, 1)); v = np.tile(state.evader_velocity, (n, 1))
        h = np.tile(d.hunter, (n, 1)); hv = np.tile(state.hunter_velocity, (n, 1))
        pending = np.tile(self.pending, (n, 1))
        obstacles = np.concatenate((d.obstacles, borders()))
        radii = np.r_[np.full(3, .05), np.full(80, .15)]
        costs = np.zeros(n); unsafe = np.zeros(n, dtype=bool)
        reached = np.zeros(n, dtype=bool)
        trajectory, first = [], None
        for t in range(cfg['horizon']):
            target = self.planner.waypoint(p, self.route, cfg['lookahead'])
            distance = np.linalg.norm(target-p, axis=1)
            speed = np.minimum(.25, distance*2)
            desired = unit(target-p)*speed[:, None]
            command = limit((desired-v)*8 + perturbations[:, t//cfg['chunk_steps']], 1.)
            if t == 0:
                first = command.copy()
            hunter_accel = unit(p-h)*.85
            if t == 0 and not self.started:
                hunter_accel[:] = 0
            pold, hold = p.copy(), h.copy()
            applied = command if self.method == 'no_delay' else pending
            p, v = advance(p, v, applied, .25, cfg['dt'], obstacles, radii)
            h, hv = advance(h, hv, hunter_accel, .28, cfg['dt'], obstacles, radii)
            pending = command
            obs_dist = segment_distance(pold[:, None, :], p[:, None, :], d.obstacles[None])
            hunt_dist = segment_distance(pold-hold, p-h, np.zeros_like(p))
            crash = (obs_dist < .079).any(axis=1) | (np.max(abs(p), axis=1) > .825)
            caught = hunt_dist < .052
            if self.method == 'no_hunter':
                caught[:] = False
            active = ~reached
            unsafe |= (crash | caught) & active
            new_success = (np.linalg.norm(p-d.goal, axis=1) < .070) & ~crash & ~caught & active
            goal_distance = self.planner.cost_to_goal(p)
            hunter_risk = np.maximum(.35-hunt_dist, 0)**2
            if self.method == 'no_hunter':
                hunter_risk[:] = 0
            costs += active*(goal_distance*.04 + hunter_risk*cfg['hunter_weight']
                      + np.sum(np.maximum(.12-obs_dist, 0)**2, axis=1)*4
                      + (crash | caught)*100)
            costs -= new_success*15
            reached |= new_success
            trajectory.append(p.copy())
        costs += np.where(reached, 0., self.planner.cost_to_goal(p)*3)
        return costs, first, np.array(trajectory), unsafe
