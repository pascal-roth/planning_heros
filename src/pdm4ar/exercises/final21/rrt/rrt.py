import time
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
from pdm4ar.exercises.final21.rrt.distance import Distance, DistanceMetric
from pdm4ar.exercises.final21.rrt.cost import euclidean_cost


class RRT:
    def __init__(self, goal: PolygonGoal,
                 static_obstacles: Sequence[StaticObstacle],
                 n_samples: int = 1000,
                 distance_metric: DistanceMetric = DistanceMetric.L2,
                 radius: Optional[float] = 5.0):
        """
        :param goal:                Goal Polygon
        :param static_obstacles:    Obstacles in the given environment, note that obstacle 0 is the boundary of the
                                    environment !!!
        :param n_samples:           point cloud with random n samples from the environment using Halton Sequence
        """

        self.goal: PolygonGoal = goal
        self.static_obstacles: Sequence[StaticObstacle] = static_obstacles

        # init sampler and point cloud
        self.n_samples: int = n_samples
        self.sampler: Sampler = Sampler(static_obstacles, n_samples=n_samples)
        # self.sampler.plot_samples(self.goal)

        # init distance calculation (atm with brute force distance and euclidean distance)
        self.radius: Optional[float] = radius
        self.distance = Distance(distance_metric)

        # TODO: init movement primitives
        # TODO: init a proper cost class, that takes time and distance into account
        self.cost_fct: Callable = euclidean_cost

        # for each point in the point_cloud introduce a SampleState including cost and parent
        self.sample_state: Dict[int, SampleState] = {}

    def plan_path(self, spacecraft_state: SpacecraftState, plot: bool = True):
        t_start = time.time()

        # add start point to point-cloud (pc)
        self._add_start_to_pc(spacecraft_state)

        # build kdtree and determine distance of each point from Spacecraft
        self.distance.init_tree(self.sampler.pc2array())
        distance = self.distance.get_distance(self.sampler.point_cloud_idx_latest-1,
                                              point_cloud=self.sampler.pc2array())
        distance_idx_sorted = np.argsort(distance)

        for idx in distance_idx_sorted:
            self._update_graph(idx)

        t_end = time.time()
        print(f'Building RRT* map in {np.round(t_end - t_start, decimals=4)}s')

        if plot:
            ax = self._init_plot()
            self._plotter(ax)

        # TODO: get optimal plan

        # TODO: reuse part of the sample state has not been changed due to moving obstacle or movement of the spacecraft
        self.sample_state = {}

    def refine_path(self, n_samples: int) -> None:
        # function not tested
        x_rand_idx = self.sampler.draw_samples(n_samples)
        self.distance.init_tree(self.sampler.pc2array())

        for idx in range(n_samples):
            self._update_graph(x_rand_idx)

    def _add_start_to_pc(self, spacecraft_state: SpacecraftState) -> None:
        x_start = np.expand_dims(np.array([spacecraft_state.x, spacecraft_state.y]), axis=0)
        self.sampler.add_sample(x_start)
        self.sample_state[self.sampler.point_cloud_idx_latest-1] = SampleState(
            parent=self.sampler.point_cloud_idx_latest-1, cost=0)

    def _update_graph(self, x_idx):
        x_rand = self.sampler.point_cloud[x_idx]

        # TODO: calculate x_new as the point closest to x_rand that can be reached with a movement primitive without
        #  collision, in the following it is assumed that a collision free path is possible
        x_new = x_rand

        # collect closest sample and all samples within distance eth=radius
        _, x_near_idx = self.distance.tree.query(x_new, k=2)
        x_near_idx = x_near_idx[1]
        X_near_idx = self.distance.tree.query_ball_point(x_new, r=self.radius, workers=-1)
        # TODO: update the radius dynamically with maximum distance that can be reached by movement primitive and
        #  that log(n) samples are included in it

        # check if x_near in the sample_state otherwise set it to start
        additional_cost = 0
        if not self.sample_state.get(x_near_idx):
            x_near_idx = self.sampler.point_cloud_idx_latest-1
            if x_near_idx not in X_near_idx:  # additional cost just necessary until movement primitives and proper cost
                additional_cost = 1000

        # init x_min and its cost
        x_min_idx = x_near_idx
        c_min = self.sample_state[x_near_idx].cost + euclidean_cost(self.sampler.point_cloud[x_near_idx], x_new) + additional_cost

        # check for all samples within radius which results in the smallest cost to reach the new sample
        for x_near_i in X_near_idx:
            if not self.sample_state.get(x_near_i):
                continue
            cost = self.sample_state[x_near_i].cost + euclidean_cost(self.sampler.point_cloud[x_near_i], x_new)
            if cost < c_min:  # TODO: also add collision check
                x_min_idx = x_near_i
                c_min = cost

        self.sample_state[x_idx] = SampleState(parent=x_min_idx, cost=c_min)

        # rebuild tree s.t. samples that can be reached with a smaller cost from the x_new are updated
        for x_near_i in X_near_idx:
            if not self.sample_state.get(x_near_i):
                continue
            cost = c_min + euclidean_cost(x_new, self.sampler.point_cloud[x_near_i])
            if cost < self.sample_state[x_near_i].cost:  # TODO: also add collision check
                self.sample_state[x_near_i].parent = x_idx

    def _init_plot(self):
        matplotlib.use('TkAgg')
        ax = plt.gca()
        shapely_viz = ShapelyViz(ax)

        for s_obstacle in self.static_obstacles:
            shapely_viz.add_shape(s_obstacle.shape, color=s_obstacle.geometry.color,
                                  zorder=ZOrders.ENV_OBSTACLE)
        shapely_viz.add_shape(self.goal.get_plottable_geometry(), color="orange", zorder=ZOrders.GOAL, alpha=0.5)
        ax = shapely_viz.ax
        ax.autoscale()
        ax.set_facecolor('k')
        ax.set_aspect("equal")
        return ax

    def _plotter(self, ax):
        pc_points = [point for point in self.sample_state.keys()]
        pc_parents = [state.parent for state in self.sample_state.values()]
        for i in range(len(pc_points)):
            pc_point = self.sampler.point_cloud[pc_points[i]]
            pc_parent = self.sampler.point_cloud[pc_parents[i]]
            ax.plot((pc_point[0], pc_parent[0]), (pc_point[1], pc_parent[1]), color='r')
        ax.set_title(f'added {self.n_samples} samples')
        plt.show()


class SampleState:
    def __init__(self, parent: int, cost: float):
        self.parent: int = parent
        self.cost: cost = cost
