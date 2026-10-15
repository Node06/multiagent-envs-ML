"""MPC shooting and recovery with learned predictions in every rollout.
Adapted from the preserved vision_mpc V1/V2 planner; no analytic-physics fallback.
"""
import numpy as np
from vision_mpc.geometry import limit, unit, segment_distance
from vision_mpc.dynamics import borders


class ShootingController:
    """Feedback shooting MPC with CEM; real environment is never used for rollouts."""
    def __init__(self, config, planner, method='mpc', world=None):
        if world is None: raise ValueError('A trained world model is required')
        self.world = world
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
        nominal = limit((unit(target-d.evader)*self.world.speed_cap(0)-state.evader_velocity)*8, 1.)
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
        p_error = np.zeros(n); h_error = np.zeros(n)
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
            speed = np.minimum(self.world.speed_cap(0), distance*2)
            desired = unit(target-p)*speed[:, None]
            command = limit((desired-v)*8 + perturbations[:, t//cfg['chunk_steps']], 1.)
            if t == 0:
                first = command.copy()
            hunter_accel = unit(p-h)*.85
            if t == 0 and not self.started:
                hunter_accel[:] = 0
            pold, hold = p.copy(), h.copy()
            applied = command if self.method == 'no_delay' else pending
            p, v, psigma = self.world.advance(p, v, applied, 0, obstacles, radii)
            h, hv, hsigma = self.world.advance(h, hv, hunter_accel, 1, obstacles, radii)
            p_error += psigma*cfg['dt']; h_error += hsigma*cfg['dt']
            pending = command
            obs_dist = segment_distance(pold[:, None, :], p[:, None, :], d.obstacles[None])
            hunt_dist = segment_distance(pold-hold, p-h, np.zeros_like(p))
            obs_dist -= cfg.get('uncertainty_scale', .25)*p_error[:, None]
            hunt_dist -= cfg.get('uncertainty_scale', .25)*(p_error+h_error)
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


class Controller(ShootingController):
    def choose(self, state):
        queued, started = self.pending.copy(), self.started
        command, debug = super().choose(state)
        debug['recovery'] = False
        if self.method != 'mpc' or not debug['unsafe']:
            return command, debug
        self.pending, self.started = queued, started
        commands, trajectories, costs, unsafe, survival = self.recovery(state)
        safe = np.flatnonzero(~unsafe)
        if len(safe):
            pick = safe[np.argmin(costs[safe])]
        else:
            longest = np.flatnonzero(survival == survival.max())
            pick = longest[np.argmin(costs[longest])]
        self.pending, self.started = commands[pick].copy(), True
        self.mean[:] = 0
        return self.pending.copy(), {'route': self.route, 'predicted': trajectories[:, pick],
                                     'unsafe': bool(unsafe[pick]), 'recovery': True,
                                     'cost': float(costs[pick]), 'predicted_survival': int(survival[pick])}

    def recovery(self, state):
        d, cfg = state.detection, self.cfg
        anchors, lengths, routes = [], [], []
        for radius in (.30, .50):
            for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
                anchor = np.clip(d.evader + radius*np.array([np.cos(angle), np.sin(angle)]), -.77, .77)
                if np.linalg.norm(anchor-d.evader) < .12 or np.linalg.norm(anchor-d.obstacles, axis=1).min() < .10:
                    continue
                route = self.planner.plan(anchor)
                for length in (4, 10, 18, 26):
                    anchors.append(anchor); lengths.append(length); routes.append(route)
        if not anchors:
            anchors = [d.evader]; lengths = [0]; routes = [self.route]
        anchors = np.array(anchors); lengths = np.array(lengths)
        n = len(anchors); horizon = cfg.get('recovery_horizon', 50)
        p_error = np.zeros(n); h_error = np.zeros(n)
        max_length = max(map(len, routes))
        padded = np.array([np.vstack((r, np.repeat(r[-1][None], max_length-len(r), axis=0))) for r in routes])
        p = np.tile(d.evader, (n, 1)); v = np.tile(state.evader_velocity, (n, 1))
        h = np.tile(d.hunter, (n, 1)); hv = np.tile(state.hunter_velocity, (n, 1))
        queued = np.tile(self.pending, (n, 1))
        obstacles = np.concatenate((d.obstacles, borders()))
        radii = np.r_[np.full(3, .05), np.full(80, .15)]
        costs = np.zeros(n); unsafe = np.zeros(n, bool); reached = np.zeros(n, bool)
        survival = np.full(n, horizon); trajectory = []; first = None
        for t in range(horizon):
            nearest = np.argmin(np.sum((p[:, None]-padded)**2, axis=-1), axis=1)
            target_index = np.minimum(nearest+int(cfg['lookahead']/self.planner.res), padded.shape[1]-1)
            path_target = padded[np.arange(n), target_index]
            target = np.where((t < lengths)[:, None], anchors, path_target)
            delta = target-p
            desired = unit(delta)*np.minimum(self.world.speed_cap(0), np.linalg.norm(delta, axis=1)*2)[:, None]
            q = limit((desired-v)*8, 1.)
            if t == 0:
                first = q.copy()
            ha = unit(p-h)*.85
            if t == 0 and not self.started:
                ha[:] = 0
            oldp, oldh = p.copy(), h.copy()
            p, v, psigma = self.world.advance(p, v, queued, 0, obstacles, radii)
            h, hv, hsigma = self.world.advance(h, hv, ha, 1, obstacles, radii)
            p_error += psigma*cfg['dt']; h_error += hsigma*cfg['dt']
            queued = q
            obstacle_distance = segment_distance(oldp[:, None], p[:, None], d.obstacles[None])
            hunter_distance = segment_distance(oldp-oldh, p-h, np.zeros_like(p))
            obstacle_distance -= cfg.get('uncertainty_scale', .25)*p_error[:, None]
            hunter_distance -= cfg.get('uncertainty_scale', .25)*(p_error+h_error)
            crash = (obstacle_distance < .079).any(axis=1) | (np.max(abs(p), axis=1) > .825)
            captured = hunter_distance < .052
            active = ~reached
            violation = (crash | captured) & active
            survival[(~unsafe) & violation] = t
            unsafe |= violation
            success = (np.linalg.norm(p-d.goal, axis=1) < .070) & ~violation & active
            costs += active*(self.planner.cost_to_goal(p)*.04 + np.maximum(.35-hunter_distance, 0)**2*cfg['hunter_weight']
                             + np.maximum(.12-obstacle_distance, 0).sum(axis=1)*.2 + violation*100)
            costs -= success*15
            reached |= success
            trajectory.append(p.copy())
        costs += np.where(reached, 0., self.planner.cost_to_goal(p)*3)
        return first, np.array(trajectory), costs, unsafe, survival
