import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Sequence, Optional, Callable, Dict

from dg_commons.planning import PolygonGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.spacecraft import SpacecraftState
from dg_commons.maps.shapely_viz import ShapelyViz
from dg_commons.sim.simulator_visualisation import ZOrders

from pdm4ar.exercises.final21.rrt.sampler import Sampler
from pdm4ar.exercises.final21.rrt.distance import Distance, DistanceMethod, DistanceMetric
from pdm4ar.exercises.final21.rrt.cost import euclidean_cost


class RRT:
    def __init__(self, goal: PolygonGoal,
                 static_obstacles: Sequence[StaticObstacle],
                 n_init: int = 1,
                 n_total: int = 1000,
                 distance_method: DistanceMethod = DistanceMethod.BF, 
                 distance_metric: DistanceMetric = DistanceMetric.L2, 
                 radius: Optional[float] = 10.0):
        """
        :param goal:                Goal Polygon
        :param static_obstacles:    Obstacles in the given environment, note that obstacle 0 is the boundary of the
                                    environment !!!
        :param n_init:              init point cloud with random n samples from the environment using Halton Sequence
        """

        self.goal: PolygonGoal = goal
        self.static_obstacles: Sequence[StaticObstacle] = static_obstacles

        # init sampler and point cloud
        self.n_init: int = n_init
        self.n_total: int = n_total
        self.sampler: Sampler = Sampler(static_obstacles, n_samples=n_init)
        self.sampler.plot_samples(self.goal)

        # init distance calculation (atm with brute force distance and euclidean distance)
        self.radius: Optional[float] = radius
        self.distance = Distance(distance_method, distance_metric)

        # TODO: init movement primitives
        # TODO: init a proper cost class, that takes time and distance into account
        self.cost_fct: Callable = euclidean_cost

        # for each point in the point_cloud introduce a SampleState including cost and parent
        self.sample_state: Dict[int, SampleState] = {}

    def plan_path(self, spacecraft_state: SpacecraftState, plot: bool = True):
        # construct a course path map for the initial point cloud, possible helpful for increasing the speed
        x_start = np.expand_dims(np.array([spacecraft_state.x, spacecraft_state.y]), axis=0)
        self.sampler.add_sample(x_start)
        x_start_idx = self.sampler.point_cloud_idx_latest-1
        self.sample_state[x_start_idx] = SampleState(parent=x_start_idx, cost=0)
        distance = self.distance._get_distance(x_start_idx, point_cloud=self.sampler.pc2array())
        distance_idx_sorted = np.argsort(distance)
        for idx in range(self.n_init):
            self._update_graph(distance_idx_sorted[idx])

        if plot:
            matplotlib.use('TkAgg')
            ax = plt.gca()
            # shapely_viz = ShapelyViz(ax)
            #
            # for s_obstacle in self.static_obstacles:
            #     shapely_viz.add_shape(s_obstacle.shape, color=s_obstacle.geometry.color,
            #                           zorder=ZOrders.ENV_OBSTACLE)
            # shapely_viz.add_shape(self.goal.get_plottable_geometry(), color="orange", zorder=ZOrders.GOAL, alpha=0.5)
            # ax = shapely_viz.ax
            ax.autoscale()
            ax.set_facecolor('k')
            ax.set_aspect("equal")

        # add points to path map until n_total is reached
        for idx in range(self.n_total - self.n_init):
            x_rand_idx = self.sampler.draw_samples()[0]
            self._update_graph(x_rand_idx)

            if plot:
                pc_points = [point for point in self.sample_state.keys()]
                pc_parents = [state.parent for state in self.sample_state.values()]
                for i in range(len(pc_points)):
                    pc_point = self.sampler.point_cloud[pc_points[i]]
                    pc_parent = self.sampler.point_cloud[pc_parents[i]]
                    ax.plot((pc_point[0], pc_parent[0]), (pc_point[1], pc_parent[1]), color='r')
                ax.set_title(f'added {idx} samples')
                plt.pause(0.05)
            plt.show()

        # TODO: get optimal plan

    def _update_graph(self, x_rand_idx):
        # NOTE: in theory, if x_new is updated from x_rand, BF has to be executed twice, however if we say its close,
        #       the approximation makes 1 BF unnecessary
        # x_near_idx = self.distance.get_nn(x_rand_idx, point_cloud=self.sampler.pc2array())

        # TODO: calculate x_new as the point closest to x_rand that can be reached with a movement primitive without
        #  collision, in the following it is assumed that a collision free path is possible
        x_new_idx = x_rand_idx

        # collect closest sample and all samples within distance eth=radius
        x_near_idx, X_near_idx = self.distance.get_nn(x_new_idx, point_cloud=self.sampler.pc2array(),
                                                      radius=self.radius)

        # init x_min and its cost
        x_min_idx = x_near_idx
        c_min = self.sample_state[x_near_idx].cost + euclidean_cost(self.sampler.point_cloud[x_near_idx],
                                                                    self.sampler.point_cloud[x_new_idx])

        # check for all samples within radius which results in the smallest cost to reach the new sample
        for x_near_current in X_near_idx:

            # if not self.sample_state.get(x_near_current):
            #     continue

            cost = self.sample_state[x_near_current].cost + euclidean_cost(self.sampler.point_cloud[x_near_current],
                                                                           self.sampler.point_cloud[x_new_idx])
            if cost < c_min:  # TODO: also add collision check
                x_min_idx = x_near_current
                c_min = cost

        self.sample_state[x_new_idx] = SampleState(parent=x_min_idx, cost=c_min)

        # rebuild tree s.t. samples that can be reached with a smaller cost from the x_new are updated
        for x_near_current in X_near_idx:

            # if not self.sample_state.get(x_near_current):
            #     continue

            cost = c_min + euclidean_cost(self.sampler.point_cloud[x_new_idx], self.sampler.point_cloud[x_near_current])

            if cost < self.sample_state[x_near_current].cost:
                self.sample_state[x_near_current].parent = x_new_idx


class SampleState:
    def __init__(self, parent: int, cost: float):
        self.parent: int = parent
        self.cost: cost = cost
