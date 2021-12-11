from typing import Sequence, List
from dg_commons.sim.models.obstacles import StaticObstacle
from shapely.geometry import Point
import numpy as np


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
            # TODO: add bounding box check
            if obstacle.shape.contains(point):
                return True
        return False

    def is_collision_free(self, pts: np.ndarray) -> np.ndarray:
        n_pts = pts.shape[0]
        mask = np.zeros((n_pts, ), dtype=bool)
        for i in range(n_pts):
            mask[i] = self._check(Point(pts[i, :]))
        return ~mask
