"""V2: preserve nominal MPC, add unrestricted recovery proposals when unsafe."""
import numpy as np
from .controller_v1 import Controller as OriginalController
from .geometry import unit, limit, segment_distance
from .dynamics import advance, borders


class Controller(OriginalController):
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
            desired = unit(delta)*np.minimum(.25, np.linalg.norm(delta, axis=1)*2)[:, None]
            q = limit((desired-v)*8, 1.)
            if t == 0:
                first = q.copy()
            ha = unit(p-h)*.85
            if t == 0 and not self.started:
                ha[:] = 0
            oldp, oldh = p.copy(), h.copy()
            p, v = advance(p, v, queued, .25, cfg['dt'], obstacles, radii)
            h, hv = advance(h, hv, ha, .28, cfg['dt'], obstacles, radii)
            queued = q
            obstacle_distance = segment_distance(oldp[:, None], p[:, None], d.obstacles[None])
            hunter_distance = segment_distance(oldp-oldh, p-h, np.zeros_like(p))
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
