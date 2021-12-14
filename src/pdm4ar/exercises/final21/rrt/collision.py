from typing import Sequence, List
from dg_commons.sim.models.obstacles import StaticObstacle
from shapely.geometry import Point
import numpy as np

from pdm4ar.exercises.final21.rrt.params import BUFFER_DISTANCE


class BoundingBox:
    def __init__(self, obstacle: StaticObstacle):
        pass


class CollisionChecker:
    def __init__(self,
                 static_obstacles: Sequence[StaticObstacle],
                 dynamic_obstacles=None):
        self.static_obstacles: List[StaticObstacle] = list(static_obstacles)
        self.dynamic_obstacles = dynamic_obstacles
        self.dist2obstacle: List[float] = []

    def is_collision_free(self, pts: np.ndarray) -> np.ndarray:
        obstacles_buffered = [obstacle.shape.buffer(BUFFER_DISTANCE) for obstacle in self.static_obstacles]
        mask = [any([obstacle.contains(Point(pt)) for obstacle in obstacles_buffered]) for pt in pts]
        mask = np.array(mask, dtype=bool)
        return ~mask

    def path_collision_free(self, pt_start: np.ndarray, pt_end: np.ndarray, pt_distance: float, idx: int):
        if pt_distance < self.dist2obstacle[idx]:
            return True

        delta_x = pt_end[0] - pt_start[0]
        delta_y = pt_end[1] - pt_start[1]
        pts_on_line = [np.array([pt_start[0] + step * delta_x, pt_start[1] + delta_y * step]) for step in
                       np.arange(0.25, 0.1, 0.25)]
        return all(self.is_collision_free(np.array(pts_on_line)))

    def obstacle_distance(self, point_cloud: np.ndarray) -> None:
        self.dist2obstacle = [np.min([obstacle.shape.distance(Point(pc_point)) for obstacle in self.static_obstacles])
                              for pc_point in point_cloud]
