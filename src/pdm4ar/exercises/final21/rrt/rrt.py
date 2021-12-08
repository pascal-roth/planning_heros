import numpy as np
from typing import Sequence
from scipy.stats import qmc

from dg_commons.planning import PolygonGoal
from dg_commons.sim.models.obstacles import StaticObstacle

from pdm4ar.exercises.final21.rrt.sampler import Sampler


class RRT:
    def __init__(self, goal: PolygonGoal,
                 static_obstacles: Sequence[StaticObstacle],
                 n_samples: int = 1000):
        """

        :param goal:                Goal Polygon
        :param static_obstacles:    Obstacles in the given environment, note that obstacle 0 is the boundary of the
                                    environment !!!
        :param n_samples:           init point cloud with random n samples from the environment using Halton Sequence
        """

        self.goal: PolygonGoal = goal
        self.static_obstacles: Sequence[StaticObstacle] = static_obstacles

        # init sampler and point cloud
        self.sampler: Sampler = Sampler(static_obstacles)
        self.point_cloud: np.ndarray = self.sampler.init_point_cloud(n_samples)
        self.sampler.plot_samples(self.point_cloud, self.goal)

        # init distance metric = init kd tree
        pass