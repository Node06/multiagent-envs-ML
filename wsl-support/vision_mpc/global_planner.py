import heapq
import numpy as np
from .geometry import segment_distance


class Planner:
    """A* route and reverse shortest-path field, built from detected pixels."""
    def __init__(self, detection, config):
        self.res = config['grid_resolution']
        self.bound = config['map_bound']
        self.axis = np.arange(-self.bound, self.bound + self.res/2, self.res)
        self.n = len(self.axis)
        xx, yy = np.meshgrid(self.axis, self.axis, indexing='ij')
        self.grid = np.stack((xx, yy), axis=-1)
        self.obstacles = detection.obstacles
        self.clearance = config['obstacle_clearance']
        self.blocked = (np.linalg.norm(self.grid[..., None, :] - self.obstacles, axis=-1)
                        < self.clearance).any(axis=-1)
        self.goal = detection.goal
        self.goal_index = self.index(self.goal)
        free = np.argwhere(~self.blocked)
        if self.blocked[self.goal_index]:
            nearest = np.argmin(np.linalg.norm(self.grid[~self.blocked] - self.goal, axis=1))
            self.goal_index = tuple(free[nearest])
        self.distance = np.full((self.n, self.n), np.inf)
        self.parent = {}
        self.distance[self.goal_index] = 0
        queue = [(0., self.goal_index)]
        while queue:
            cost, node = heapq.heappop(queue)
            if cost > self.distance[node]:
                continue
            for neighbor, edge in self.neighbors(node):
                new = cost + edge
                if new < self.distance[neighbor]:
                    self.distance[neighbor] = new
                    self.parent[neighbor] = node
                    heapq.heappush(queue, (new, neighbor))

    def index(self, position):
        return tuple(np.clip(np.rint((position+self.bound)/self.res).astype(int), 0, self.n-1))

    def neighbors(self, node):
        x, y = node
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if not (dx or dy):
                    continue
                nx, ny = x+dx, y+dy
                if not (0 <= nx < self.n and 0 <= ny < self.n) or self.blocked[nx, ny]:
                    continue
                if dx and dy and (self.blocked[nx, y] or self.blocked[x, ny]):
                    continue
                yield (nx, ny), self.res*np.hypot(dx, dy)

    def plan(self, start):
        begin = self.index(start)
        if self.blocked[begin] or not np.isfinite(self.distance[begin]):
            return np.array([start, self.goal])
        # Forward A*, with the exact reverse cost field as an admissible heuristic.
        queue = [(self.distance[begin], 0., begin)]
        previous, costs = {}, {begin: 0.}
        while queue:
            _, cost, node = heapq.heappop(queue)
            if cost > costs[node]:
                continue
            if node == self.goal_index:
                route = [node]
                while route[-1] != begin:
                    route.append(previous[route[-1]])
                route.reverse()
                points = np.array([self.grid[p] for p in route])
                points[0] = start
                if np.linalg.norm(points[-1]-self.goal) < .075:
                    points[-1] = self.goal
                return points
            for neighbor, edge in self.neighbors(node):
                new = cost+edge
                if new < costs.get(neighbor, np.inf):
                    previous[neighbor], costs[neighbor] = node, new
                    heapq.heappush(queue, (new+self.distance[neighbor], new, neighbor))
        return np.array([start, self.goal])

    def cost_to_goal(self, positions):
        ix = np.clip(np.rint((positions+self.bound)/self.res).astype(int), 0, self.n-1)
        value = self.distance[ix[..., 0], ix[..., 1]]
        return np.where(np.isfinite(value), value, 5.)

    def waypoint(self, positions, route, lookahead):
        nearest = np.argmin(np.sum((positions[:, None, :]-route[None, :, :])**2, axis=-1), axis=1)
        steps = max(1, int(lookahead / self.res))
        return route[np.minimum(nearest+steps, len(route)-1)]
