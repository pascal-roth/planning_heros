import time
from dg_commons.sim.models.spacecraft_structures import SpacecraftGeometry
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Sequence, Optional, Callable, Dict, List

from dg_commons.planning import PolygonGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.spacecraft import SpacecraftState
from dg_commons.maps.shapely_viz import ShapelyViz
from dg_commons.sim.simulator_visualisation import ZOrders
from pdm4ar.exercises.final21.rrt.motion_primitives import MotionPrimitives, SpacecraftTrajectory
from pdm4ar.exercises.final21.rrt.params import MOTION_PRIMITIVE_INPUT_DIVISIONS, STEERING_MAX_DIST

from pdm4ar.exercises.final21.rrt.sampler import Sampler
from pdm4ar.exercises.final21.rrt.distance import Distance, DistanceMetric
from pdm4ar.exercises.final21.rrt.cost import euclidean_cost

import networkx as nx


class Node:
    def __init__(self, state: SpacecraftState, cost: float):
        self.state = state
        self.cost: cost = cost

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.state.x, self.state.y])

    @property
    def vel(self) -> np.ndarray:
        return np.array([self.state.vx, self.state.vy])


class RRT:
    def __init__(self,
                 goal: PolygonGoal,
                 static_obstacles: Sequence[StaticObstacle],
                 sg: SpacecraftGeometry,
                 n_samples: int =1000,
                 distance_metric: DistanceMetric = DistanceMetric.L2,
                 radius: Optional[float] = 30):
        """
        :param goal:                Goal Polygon
        :param static_obstacles:    Obstacles in the given environment, note that obstacle 0 is the boundary of the
                                    environment !!!
        :param n_samples:           point cloud with random n samples from the environment using Halton Sequence
        """

        self.goal: PolygonGoal = goal
        self.static_obstacles: Sequence[StaticObstacle] = static_obstacles
        self.sg = sg

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

        self.tree = nx.DiGraph()

        self.motion_primitives = MotionPrimitives(
            self.sg, MOTION_PRIMITIVE_INPUT_DIVISIONS)
        self.radius = self.motion_primitives.max_distance_covered

    def plan_path(self, spacecraft_state: SpacecraftState, plot: bool = True):
        t_start = time.time()

        # add start point to point-cloud (pc)
        root_idx = self._add_root(spacecraft_state)

        # build kdtree and determine distance of each point from Spacecraft
        self.distance.init_tree(self.sampler.pc2array())
        distance = self.distance.get_distance(
            root_idx, point_cloud=self.sampler.pc2array())
        distance_idx_sorted = np.argsort(distance)

        for i in distance_idx_sorted:
            self._update_graph(i)

        print(f'Building RRT* map in {time.time() - t_start:.2f}s')

        if plot:
            ax = self._draw_obstacles()
            self._plotter(ax)

        # TODO: get optimal plan

        # TODO: reuse part of the sample state has not been changed due to moving obstacle or movement of the spacecraft
        # self.tree = {}

    def refine_path(self, n_samples: int) -> None:
        # function not tested
        x_rand_idx = self.sampler.draw_samples(n_samples)
        self.distance.init_tree(self.sampler.pc2array())

        for idx in range(n_samples):
            self._update_graph(x_rand_idx)

    def _add_root(self, spacecraft_state: SpacecraftState) -> int:
        x_start = np.expand_dims(np.array(
            [spacecraft_state.x, spacecraft_state.y]),
                                 axis=0)
        self.tree.add_node(Node(cost=0., state=spacecraft_state))
        root_idx = self.sampler.point_cloud_idx_latest
        self.sampler.add_sample(x_start)
        return root_idx

    def _update_graph(self, x_idx):
        x_rand = self.sampler.point_cloud[x_idx]

        # calculate x_new as the point closest to x_rand that can be reached with a movement primitive without
        # collision, in the following it is assumed that a collision free path is possible
        x_nearest = self._nearest_on_tree(x_rand)
        # if np.linalg.norm(x_nearest.pos - x_rand) > self.radius:
        #     return
        goal_state = SpacecraftState(x=x_rand[0],
                                     y=x_rand[1],
                                     psi=0,
                                     vx=0,
                                     vy=0,
                                     dpsi=0)
        x_new_dist, trajectory = self.motion_primitives.distance(
            x_nearest.state, goal_state)
        dist = np.linalg.norm(x_rand - np.array([trajectory.states[-1].x, trajectory.states[-1].y]))
        if dist > STEERING_MAX_DIST:
            # print(f"rejected: x_new_dist: {x_new_dist}")
            return
        print(x_nearest.state)
        print(trajectory.states[0])
        command = trajectory.commands[-1]
        print(f"command: {command}")
        x_new = Node(state=trajectory.states[-1],
                     cost=x_nearest.cost + trajectory.get_cost())

        # collect closest sample and all samples within distance eth=radius
        # TODO: update the radius dynamically with maximum distance that can be reached by movement primitive and
        #  that log(n) samples are included in it
        near_nodes = self._near_on_tree(
            x_new, self.radius, lambda n1, n2: self.motion_primitives.distance(
                n1.state, n2.state)[0])

        self.tree.add_node(x_new)

        # check if x_near in the tree otherwise set it to start
        # additional_cost = 0
        # if not self.tree.get(x_near_idx):
        #     x_near_idx = self.sampler.point_cloud_idx_latest - 1
        #     if x_near_idx not in X_near_idx:  # additional cost just necessary until movement primitives and proper cost
        #         additional_cost = 1000

        # init x_min and its cost
        x_min = x_nearest
        c_min = x_new.cost
        min_trajectory = trajectory
        # check for all samples within radius which results in the smallest cost to reach the new sample
        for x_near in near_nodes:
            collision_free = True
            line_cost, min_trajectory = self.motion_primitives.distance(
                x_near.state, x_new.state)
            if collision_free and x_near.cost + line_cost < c_min:
                x_min = x_near
                c_min = x_near.cost + line_cost

        self.tree.add_edge(x_min, x_new, trajectory=min_trajectory)
        
        # ax = self._draw_obstacles()
        # self._plotter(ax)
        # rebuild tree s.t. samples that can be reached with a smaller cost from the x_new are updated
        for x_near in near_nodes:
            # TODO: also add collision check
            collision_free = True
            motion_cost, trajectory = self.motion_primitives.distance(
                x_new.state, x_near.state)
            line_cost = c_min + motion_cost
            if c_min + motion_cost < x_near.cost:
                x_parent = self.tree.predecessors(x_near)
                # TODO: somehow without this check we get an error, investigate why
                if self.tree.has_edge(x_parent, x_near):
                    self.tree.remove_edge(x_parent, x_near)
                self.tree.add_edge(x_new, x_near, trajectory=trajectory)

    def _nearest_on_tree(self, pos: np.ndarray) -> Node:
        # TODO replace brute force distance search
        min_dist = np.inf
        nearest = None
        for node in self.tree.nodes:
            dist = self.distance.dist(node.pos, pos)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        return nearest

    def _near_on_tree(self, center: Node, radius: float,
                      metric: Callable[[Node, Node], float]) -> List[Node]:
        # TODO replace brute force distance search
        near: List[Node] = []
        for node in self.tree.nodes:
            if metric(node, center) < radius:
                near.append(node)
        return near

    def _draw_obstacles(self):
        matplotlib.use('TkAgg')
        ax = plt.gca()
        shapely_viz = ShapelyViz(ax)

        for s_obstacle in self.static_obstacles:
            shapely_viz.add_shape(s_obstacle.shape,
                                  color=s_obstacle.geometry.color,
                                  zorder=ZOrders.ENV_OBSTACLE)
        shapely_viz.add_shape(self.goal.get_plottable_geometry(),
                              color="orange",
                              zorder=ZOrders.GOAL,
                              alpha=0.5)
        ax = shapely_viz.ax
        ax.autoscale()
        ax.set_facecolor('k')
        ax.set_aspect("equal")
        return ax

    def _plotter(self, ax, draw_labels=False):
        pos = {node: node.pos for node in self.tree.nodes}
        nx.draw_networkx_nodes(self.tree, pos, node_size=100)
        edge_labels = dict()
        for from_node, to_node, data in self.tree.edges(data=True):
            if "trajectory" in data:
                trajectory: SpacecraftTrajectory = data["trajectory"]
                traj_pos= np.array([[s.x, s.y] for s in trajectory.states])
                plt.plot(traj_pos[:, 0], traj_pos[:, 1], color="orange", zorder=100)
                command = trajectory.commands[-1]
                edge_labels[(from_node, to_node)] = f"({command.acc_left:.1f}, {command.acc_right:.1f})"
        if draw_labels:
            nx.draw_networkx_edge_labels(self.tree, pos=pos, edge_labels=edge_labels)
        ax.set_title(f'added {self.n_samples} samples')
        plt.show()
