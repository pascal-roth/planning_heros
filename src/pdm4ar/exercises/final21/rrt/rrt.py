import numpy as np
from typing import Sequence, Optional, Callable, Dict
from scipy.stats import qmc

from dg_commons.planning import PolygonGoal
from dg_commons.sim.models.obstacles import StaticObstacle

from pdm4ar.exercises.final21.rrt.sampler import Sampler
from pdm4ar.exercises.final21.rrt.distance import Distance, DistanceMethod, DistanceMetric
from pdm4ar.exercises.final21.rrt.cost import euclidean_cost


class RRT:
    def __init__(self, goal: PolygonGoal,
                 static_obstacles: Sequence[StaticObstacle],
                 n_samples: int = 1000, 
                 distance_method: DistanceMethod = DistanceMethod.BF, 
                 distance_metric: DistanceMetric = DistanceMetric.L2, 
                 radius: Optional[float] = 10.0):
        """

        :param goal:                Goal Polygon
        :param static_obstacles:    Obstacles in the given environment, note that obstacle 0 is the boundary of the
                                    environment !!!
        :param n_samples:           init point cloud with random n samples from the environment using Halton Sequence
        """

        self.goal: PolygonGoal = goal
        self.static_obstacles: Sequence[StaticObstacle] = static_obstacles

        # init sampler and point cloud
        self.sampler: Sampler = Sampler(static_obstacles, n_samples=n_samples)
        # TODO: change pointcloud to be of type list -> faster computation
        self.sampler.plot_samples(self.goal)

        # init distance calculation (atm with brute force distance and euclidean distance)
        self.radius: Optional[float] = radius
        self.distance = Distance(distance_method, distance_metric)

        # TODO: init movement primitives
        # TODO: init a proper cost class, that takes time and distance into account
        self.cost_fct: Callable = euclidean_cost

    def plan_path(self):
        # TODO: enforce constrain that at the beginning drawn points are not far from original point, i.e. use
        #  original point cloud
        x_rand_idx = self.sampler.draw_additional_samples()
        assert len(x_rand_idx) == 1
        x_near_idx = self.distance.get_nn(self.sampler.point_cloud[x_rand_idx[0]], point_cloud=self.sampler.pc2array())

        # TODO: calculate x_new as the point closest to x_rand that can be reached with a movement primitive without
        #  collision, in the following it is assumed that a collision free path is possible
        x_new_idx = x_rand_idx

        # collect all samples within distance eth=radius
        X_near_idx = self.distance.get_nn(self.sampler.point_cloud[x_new_idx[0]], point_cloud=self.sampler.pc2array(),
                                      radius=self.radius)

        # init x_min and its cost
        x_min_idx = x_near_idx
        c_min = 0



class SampleState:
    def __init__(self, parent: np.ndarray, cost: float):
        self.parent: np.ndarray = parent
        self.cost: cost = cost
