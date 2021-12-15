from typing import Sequence, List, Tuple
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.spacecraft_structures import SpacecraftGeometry
from scipy.spatial.kdtree import KDTree
from shapely.geometry import Point
import numpy as np

from pdm4ar.exercises.final21.rrt.params import BUFFER_DISTANCE, WORLD_SIZE


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
        self.obstacles_buffered = [
            obstacle.shape.buffer(BUFFER_DISTANCE)
            for obstacle in self.static_obstacles
        ]
        boundary_pts = []
        for obstacle in self.obstacles_buffered:
            boundary = obstacle.boundary
            pts = [boundary.interpolate(offset) for offset in np.arange(0,boundary.length, 0.5)]
            for point in pts:
                boundary_pts.append([point.x, point.y])
        self.boundary_pts = np.array(boundary_pts)
        self.boundary_tree = KDTree(self.boundary_pts)
        self.spacecraft_geometry = SpacecraftGeometry.default()

    def is_collision_free(self, pts: np.ndarray) -> np.ndarray:
        # going outside the world is considered a collision
        colliding = (pts[:, 0] < 0) | (pts[:, 0] > WORLD_SIZE[0]) | (
            pts[:, 1] < 0) | (pts[:, 1] > WORLD_SIZE[1])
        colliding = colliding | np.array([
            any((obstacle.contains(Point(pt))
                 for obstacle in self.obstacles_buffered)) for pt in pts
        ],
                                         dtype=bool)
        return ~colliding

    def collding(
        self,
        pts: np.ndarray,
    ) -> Tuple[bool, float]:
        outside_world = np.asarray(
            (pts[:, 0] < 0) | (pts[:, 0] > WORLD_SIZE[0]) | (pts[:, 1] < 0) |
            (pts[:, 1] > WORLD_SIZE[1]),
            dtype=bool)
        distances, _ = self.boundary_tree.query(pts)
        return np.any(outside_world | np.asarray(
            distances <= self.spacecraft_geometry.w_half, dtype=bool)), np.min(
                distances)

    def path_collision_free(self, pt_start: np.ndarray, pt_end: np.ndarray,
                            pt_distance: float, idx: int):
        if pt_distance < self.dist2obstacle[idx]:
            return True

        delta_x = pt_end[0] - pt_start[0]
        delta_y = pt_end[1] - pt_start[1]
        pts_on_line = [
            np.array(
                [pt_start[0] + step * delta_x, pt_start[1] + delta_y * step])
            for step in np.arange(0.25, 1, 0.25)
        ]
        return all(self.is_collision_free(np.array(pts_on_line)))

    def obstacle_distance(self, point_cloud: np.ndarray) -> None:
        self.dist2obstacle = [
            np.min([
                obstacle.shape.distance(Point(pc_point))
                for obstacle in self.static_obstacles
            ]) for pc_point in point_cloud
        ]
