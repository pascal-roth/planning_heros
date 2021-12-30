from typing import Sequence, List, Tuple, Optional
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.spacecraft import SpacecraftState
from dg_commons.sim.models.spacecraft_structures import SpacecraftGeometry
from scipy.spatial.kdtree import KDTree
from shapely.geometry import Point
import numpy as np
from pdm4ar.exercises.final21.rrt.dynamic import DynamicObstacleSimulator

from pdm4ar.exercises.final21.rrt.params import DYNAMIC_BUFFER_DISTANCE, STATIC_BUFFER_DISTANCE, WORLD_SIZE


class CollisionChecker:
    def __init__(self,
                 static_obstacles: Sequence[StaticObstacle],
                 dynamic_simulator: DynamicObstacleSimulator = None,
                 horizon: Optional[float] = None):
        self.static_obstacles: List[StaticObstacle] = list(static_obstacles)
        self.dynamic_simulator = dynamic_simulator
        self.dist2obstacle: List[float] = []

        # buffer (extend) static obstacles to keep the spacecraft away from them
        self.obstacle_shapes = [
            obstacle.shape.buffer(STATIC_BUFFER_DISTANCE)
            for obstacle in self.static_obstacles
        ]
        if dynamic_simulator is not None:
            # add the predicted occupancies of all dynamic obstacles as obstacles to avoid
            self.obstacle_shapes += [
                occupancy.buffer(DYNAMIC_BUFFER_DISTANCE) for name, occupancy
                in dynamic_simulator.compute_occupancies(horizon).items()
            ]

        boundary_pts = []
        for obstacle in self.obstacle_shapes:
            boundary = obstacle.boundary
            pts = [
                boundary.interpolate(offset)
                for offset in np.arange(0, boundary.length, 0.5)
            ]
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
            any((shape.contains(Point(pt)) for shape in self.obstacle_shapes))
            for pt in pts
        ],
                                         dtype=bool)
        return ~colliding

    def collding(self, states: List[SpacecraftState]) -> Tuple[bool, float]:
        pos = np.array([[state.x, state.y] for state in states])

        outside_world = np.asarray(
            (pos[:, 0] < 0) | (pos[:, 0] > WORLD_SIZE[0]) | (pos[:, 1] < 0) |
            (pos[:, 1] > WORLD_SIZE[1]),
            dtype=bool)
        distances, _ = self.boundary_tree.query(pos)
        max_spacecraft_size = self.spacecraft_geometry.w_half
        return np.any(outside_world
                      | np.asarray(distances <= max_spacecraft_size,
                                   dtype=bool)), np.min(distances)

    def path_collision_free(self, pt_start: np.ndarray, pt_end: np.ndarray,
                            pt_distance: float, idx: int):
        # if pt_distance < self.dist2obstacle[idx]:
        #     return True

        delta_x = pt_end[0] - pt_start[0]
        delta_y = pt_end[1] - pt_start[1]
        pts_on_line = [
            np.array([
                pt_start[0] + offset * delta_x, pt_start[1] + offset * delta_y
            ]) for offset in np.linspace(0, 1, 4)
        ]
        return np.all(self.is_collision_free(np.array(pts_on_line)))

    def obstacle_distance(self, point_cloud: np.ndarray) -> None:
        self.dist2obstacle = [
            np.min([
                shape.distance(Point(pc_point))
                for shape in self.obstacle_shapes
            ]) for pc_point in point_cloud
        ]

    def bounding_box_at(self, state: SpacecraftState) -> np.ndarray:
        s = self.spacecraft_geometry
        s_psi = np.sin(state.psi)
        c_psi = np.cos(state.psi)
        # spacecraft bounding box with CoG at origin
        bounding_box = np.array([[-s.lr, s.w_half], [s.lf, s.w_half],
                                 [s.lf, -s.w_half], [-s.lr, -s.w_half]])
        # rotate by state.psi
        rotation = np.array([[c_psi, -s_psi], [s_psi, c_psi]])
        bounding_box = (rotation @ bounding_box.T).T
        # move to state.pos
        bounding_box += np.array([state.x, state.y])
        return bounding_box
