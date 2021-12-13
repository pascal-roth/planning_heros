import time
from dg_commons.sim.models.spacecraft_structures import SpacecraftGeometry
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Sequence, Optional, Callable, Dict, List
from shapely.geometry import Point, LineString

from dg_commons.planning import PolygonGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.spacecraft import SpacecraftState
from dg_commons.maps.shapely_viz import ShapelyViz
from dg_commons.sim.simulator_visualisation import ZOrders

from pdm4ar.exercises.final21.rrt.motion_primitives import MotionPrimitives, SpacecraftTrajectory
from pdm4ar.exercises.final21.rrt.params import MAX_GOAL_VEL, MOTION_PRIMITIVE_INPUT_DIVISIONS, STEERING_MAX_DIST
from pdm4ar.exercises.final21.rrt.sampler import Sampler
from pdm4ar.exercises.final21.rrt.distance import Distance, DistanceMetric
from pdm4ar.exercises.final21.rrt.cost import euclidean_cost

import networkx as nx
import heapq
from collections import defaultdict


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
                 n_samples: int = 1000,
                 distance_metric: DistanceMetric = DistanceMetric.L2,
                 radius: Optional[float] = 10):
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
        self.distance = Distance(distance_metric)

        self.cost_fct: Callable = euclidean_cost

        self.tree = nx.DiGraph()
        self.tree_idx: Dict[int, Node] = {}
        self.motion_primitives = MotionPrimitives(
            self.sg, MOTION_PRIMITIVE_INPUT_DIVISIONS)
        self.radius = self.motion_primitives.max_distance_covered

    def plan_path(self, spacecraft_state: SpacecraftState):
        rrt_path = self.plan_rrt_path(spacecraft_state=spacecraft_state)
        motion_path = self.plan_motion_path(spacecraft_state, rrt_path)
        # plot
        ax = self._draw_obstacles()
        # plot rrt_path
        pos = np.array([node.pos for node in rrt_path])
        plt.plot(pos[:, 0], pos[:, 1], color="red", zorder=100)

        # plot motion path
        for trajectory in motion_path:
            if trajectory is not None:
                pos = np.array([[s.x, s.y] for s in trajectory.states])
                plt.plot(pos[:, 0], pos[:, 1], color="orange", zorder=100)
        plt.show()
        return motion_path

    def plan_rrt_path(self,
                      spacecraft_state: SpacecraftState,
                      plot: bool = True) -> List[Node]:
        t_start = time.time()

        # add start point to point-cloud (pc)
        root_idx = self._add_root(spacecraft_state)

        # build kdtree and determine distance of each point from Spacecraft
        self.distance.init_tree(self.sampler.pc2array())
        distance = self.distance.get_distance(
            root_idx, point_cloud=self.sampler.pc2array())
        distance_idx_sorted = np.argsort(distance)

        for idx in distance_idx_sorted:
            self._update_graph(idx)

        print(f'Building RRT* map in {time.time() - t_start:.2f}s')

        if plot:
            ax = self._draw_obstacles()
            self._plotter(ax)

        return self._get_optimal_rrt_path(plot)

    def plan_motion_path(self, start: SpacecraftState, rrt_path: List[Node]):
        rrt_line = LineString([node.pos for node in rrt_path])
        # A*
        state = start
        costs = defaultdict(lambda: np.inf)
        costs[state] = 0
        parents = dict()
        frontier = []
        primitives = dict()
        primitives[state] = None
        heapq.heappush(frontier, (costs[state], state))

        def is_goal(state: SpacecraftState) -> bool:
            in_goal = self.goal.goal.contains(Point([state.x, state.y]))
            speed = np.linalg.norm([state.vx, state.vy])
            slow =  speed < MAX_GOAL_VEL
            if is_goal and not slow:
                print(f"too fast: {speed}")
            return in_goal and slow

        def cost(primitive: SpacecraftTrajectory) -> float:
            primitive_pos = np.array([[state.x, state.y]
                                      for state in primitive.states])
            projected = [
                rrt_line.interpolate(
                    rrt_line.project(Point([state.x, state.y])))
                for state in primitive.states
            ]
            projected_np = np.array([[point.x, point.y]
                                     for point in projected])
            norms = np.linalg.norm(projected_np - primitive_pos, axis=1)
            return np.max(norms)

        def heuristic(state: SpacecraftState) -> float:
            return np.linalg.norm(rrt_path[-1].pos -
                                  np.array([state.x, state.y]))

        while len(frontier) > 0:
            prio, state = heapq.heappop(frontier)
            goal_dist = np.linalg.norm(rrt_path[-1].pos -
                                       np.array([state.x, state.y]))
            print(prio, goal_dist)
            if is_goal(state):
                path = []
                p = state
                while True:
                    try:
                        path.append(p)
                        p = parents[p]
                    except KeyError:
                        break
                motion_path = []
                for p in reversed(path):
                    motion_path.append(primitives[p])
                return motion_path
            for primitive in self.motion_primitives.get_primitives_from(state):
                end_state = primitive.states[-1]
                new_cost = costs[state] + cost(primitive)
                primitives[end_state] = primitive
                if new_cost < costs[end_state]:
                    costs[end_state] = new_cost
                    parents[end_state] = state
                    heapq.heappush(
                        frontier, (new_cost + heuristic(end_state), end_state))
        return []

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
        root_idx = self.sampler.point_cloud_idx_latest
        node = Node(cost=0., state=spacecraft_state)
        self.tree.add_node(node)
        self.sampler.add_sample(x_start)
        self.tree_idx[root_idx] = node
        return root_idx

    def _update_graph(self, x_idx):
        x_rand = self.sampler.point_cloud[x_idx]
        X_near_idx = self.distance.tree.query_ball_point(x_rand,
                                                         r=self.radius,
                                                         workers=-1)
        X_near_mask = [
            x_near_idx in self.tree_idx.keys() for x_near_idx in X_near_idx
        ]
        X_near_idx_prune = [
            X_near_idx[idx] for idx, x_mask in enumerate(X_near_mask) if x_mask
        ]

        # if not sample close to x_rand in the tree, terminate update
        if not X_near_idx_prune:
            return

        x_nearest = self.tree_idx[X_near_idx_prune[0]]
        goal_state = SpacecraftState(x=x_rand[0],
                                     y=x_rand[1],
                                     psi=0,
                                     vx=0,
                                     vy=0,
                                     dpsi=0)

        # add new node to tree
        x_new = Node(state=goal_state,
                     cost=x_nearest.cost +
                     self.cost_fct(x_nearest.pos, x_rand))
        self.tree.add_node(x_new)
        self.tree_idx[x_idx] = x_new

        # collect the samples within range, if no tree points in the radius, terminate update
        if len(X_near_idx_prune) > 1:
            near_nodes = [self.tree_idx[idx] for idx in X_near_idx_prune[1:]]
        else:
            return

        # init x_min and its cost
        x_min = x_nearest
        c_min = x_new.cost

        # check for all samples within radius which results in the smallest cost to reach the new sample
        for x_near in near_nodes:
            collision_free = True  # TODO: also add collision check
            line_cost = self.cost_fct(x_near.pos, x_rand)
            if collision_free and x_near.cost + line_cost < c_min:
                x_min = x_near
                c_min = x_near.cost + line_cost
        self.tree.add_edge(x_min,
                           x_new,
                           trajectory=SpacecraftTrajectory(
                               [], [x_min.state, x_new.state], 0, 0, 0))

        # rebuild tree s.t. samples that can be reached with a smaller cost from the x_new are updated
        for x_near in near_nodes:
            # TODO: also add collision check
            collision_free = True
            motion_cost = c_min + self.cost_fct(x_rand,
                                                x_near.pos)  # motion_cost
            if c_min + motion_cost < x_near.cost:
                x_parent = self.tree.predecessors(x_near)
                self.tree.remove_edge(next(x_parent), x_near)
                self.tree.add_edge(x_new,
                                   x_near,
                                   trajectory=SpacecraftTrajectory(
                                       [], [x_new.state, x_near.state], 0, 0,
                                       0))

    def _get_optimal_rrt_path(self, plot: bool = True) -> List[Node]:
        # get all points in the goal region
        sample_mask = [
            self.goal.goal.contains(Point(pc_point))
            for pc_point in self.sampler.point_cloud.values()
        ]
        goal_nodes = [
            self.tree_idx[idx] for idx in self.tree_idx.keys()
            if sample_mask[idx]
        ]
        goal_costs = [node.cost for node in goal_nodes]
        goal_node = goal_nodes[np.argmin(goal_costs)]

        rrt_path = []
        parent = goal_node
        while True:
            try:
                current_node = parent
                rrt_path.append(current_node)
                parent = next(self.tree.predecessors(current_node))
            except StopIteration:
                break
        rrt_path.reverse()
        return rrt_path

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

    def _plotter(self, ax):
        pos = {node: node.pos for node in self.tree.nodes}
        nx.draw_networkx_nodes(self.tree, pos, node_size=100)
        for from_node, to_node, data in self.tree.edges(data=True):
            if "trajectory" in data:
                trajectory: SpacecraftTrajectory = data["trajectory"]
                pos = np.array([[s.x, s.y] for s in trajectory.states])
                plt.plot(pos[:, 0], pos[:, 1], color="orange", zorder=100)
        ax.set_title(f'added {self.n_samples} samples')
        plt.show()
