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
        # self.bounding_boxes = [for obstacle in sta]

    def _check(self, point: Point) -> bool:
        for obstacle in self.static_obstacles:
            # check bounding box
            obstacle_buffered = obstacle.shape.buffer(BUFFER_DISTANCE)
            if obstacle_buffered.contains(point):
                return True
        return False

    def is_collision_free(self, pts: np.ndarray) -> np.ndarray:
        n_pts = pts.shape[0]
        mask = np.zeros((n_pts, ), dtype=bool)
        for i in range(n_pts):
            mask[i] = self._check(Point(pts[i, :]))
        return ~mask

    def path_collision_free(self, pt_start: np.ndarray, pt_end: np.ndarray):
        delta_x = pt_end[0] - pt_start[0]
        delta_y = pt_end[1] - pt_start[1]
        pts_on_line = [np.array([pt_start[0] + step * delta_x, pt_start[1] + delta_y * step]) for step in
                       np.arange(0.25, 1, 0.25)]
        return all(self.is_collision_free(np.array(pts_on_line)))

