from os import device_encoding
import time
from dg_commons.sim.models.spacecraft_structures import SpacecraftGeometry, SpacecraftParameters
from dg_commons.sim.simulator_structures import PlayerObservations
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Sequence, Optional, Callable, Dict, List, Tuple
from shapely.coords import CoordinateSequence
from shapely.geometry import Point, LineString
import networkx as nx
import heapq
from collections import defaultdict

from dg_commons.planning import PolygonGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.spacecraft import SpacecraftCommands, SpacecraftModel, SpacecraftState
from dg_commons.maps.shapely_viz import ShapelyViz
from dg_commons.sim.simulator_visualisation import ZOrders
from pdm4ar.exercises.final21.rrt.collision import CollisionChecker
from pdm4ar.exercises.final21.rrt.dynamic import DynamicObstacleSimulator

from pdm4ar.exercises.final21.rrt.motion_primitives import MotionPrimitives, SpacecraftTrajectory
from pdm4ar.exercises.final21.rrt.params import MAX_GOAL_VEL, MOTION_PRIMITIVE_INPUT_DIVISIONS, MIN_CURVATURE, \
    PRUNE_ITERATIONS
from pdm4ar.exercises.final21.rrt.sampler import Sampler
from pdm4ar.exercises.final21.rrt.distance import Distance, DistanceMetric
from pdm4ar.exercises.final21.rrt.cost import euclidean_cost

# use plots in developmend, turn off for simulation
plot = False


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
                 n_samples: int = 500,
                 distance_metric: DistanceMetric = DistanceMetric.L2,
                 radius: Optional[float] = None):
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
        if plot:
            self.sampler.plot_samples(self.goal)

        # init distance calculation (atm with brute force distance and euclidean distance)
        self.distance = Distance(distance_metric)

        self.cost_fct: Callable = euclidean_cost

        self.tree = nx.DiGraph()
        self.tree_idx: Dict[int, Node] = {}
        self.motion_primitives = MotionPrimitives(
            self.sg, MOTION_PRIMITIVE_INPUT_DIVISIONS)
        self.collision_checker: CollisionChecker = None
        self.dynamic_simulator: DynamicObstacleSimulator = None

        if radius is None:
            self.radius: float = np.sqrt(1 / np.pi * np.log(n_samples) *
                                         ((10**4) / n_samples))
            print(f'Radius selected to be: ', np.round(self.radius,
                                                       decimals=3))
        else:
            self.radius: float = radius

    def plan_path(
        self,
        spacecraft_state: SpacecraftState,
        other_players: Dict[str, PlayerObservations] = None
    ) -> Callable[[float], SpacecraftCommands]:
        # initiate dynamic obstacle simulator if dynamic obstacles are present
        if other_players is not None and len(other_players) > 0:
            self.dynamic_simulator = DynamicObstacleSimulator(
                self.sg, other_players)
        self.collision_checker = CollisionChecker(self.static_obstacles,
                                                  self.dynamic_simulator)

        t_start = time.time()
        rrt_path = self.plan_rrt_path(spacecraft_state=spacecraft_state)
        t_rrt = time.time() - t_start
        t_start = time.time()
        motion_path, primitives = self.plan_motion_path(
            spacecraft_state, rrt_path)
        t_motion = time.time() - t_start
        print(
            f'Planned motion path, total: {t_rrt + t_motion:.2f}s, rrt: {t_rrt:.2f}s, motion: {t_motion:.2f}s'
        )
        print(f"motion path goal state: {motion_path[-1]}")
        # clear tree_idx and tree
        self.tree_idx = {}
        self.tree = nx.DiGraph()

        def policy(time: float) -> SpacecraftCommands:
            assert time >= 0
            t = 0
            for trajectory in motion_path:
                t += trajectory.tf
                if t > time:
                    break
            l = trajectory.commands[0].acc_left
            r = trajectory.commands[0].acc_right
            return SpacecraftCommands(l, r)

        if plot:
            # plot
            matplotlib.use('TkAgg')
            f, (ax1, ax2) = plt.subplots(1, 2)
            self._draw_obstacles(ax1)
            # plot rrt_path
            pos = np.array([node.pos for node in rrt_path])
            ax1.plot(pos[:, 0],
                     pos[:, 1],
                     color="red",
                     zorder=100,
                     marker='*',
                     label="RRT path")

            # plot motion path
            pos = []
            for trajectory in motion_path:
                for state in trajectory.states:
                    pos.append([state.x, state.y])
            pos = np.array(pos)
            ax1.plot(pos[:, 0],
                     pos[:, 1],
                     zorder=90,
                     label="motion path",
                     color="orange")
            # pos = []
            # for trajectory in primitives.values():
            #     if trajectory:
            #         pos = np.array([[state.x, state.y]
            #                         for state in trajectory.states])
            #         ax1.plot(pos[:, 0], pos[:, 1], alpha=0.1, color="gray")

            # plot policy path
            total_time = np.sum([trajectory.tf for trajectory in motion_path])
            # self.policy_path = self.motion_primitives.test_dynamics(spacecraft_state, policy, total_time)
            policy_path_ts, self.policy_path = self.test_dynamics(
                spacecraft_state, policy, total_time)
            policy_pos = np.array([[state.x, state.y]
                                   for state in self.policy_path])
            ax1.plot(policy_pos[:, 0],
                     policy_pos[:, 1],
                     zorder=90,
                     label="policy_path",
                     color="cyan")

            ax1.legend()

            t = 0
            acc = []
            for trajectory in motion_path:
                l = trajectory.commands[0].acc_left
                r = trajectory.commands[0].acc_right
                acc.append([t, l, r])
                for i, state in enumerate(trajectory.states):
                    t_sub = t + (i / len(trajectory.states)) * trajectory.tf
                    acc.append([t_sub, *policy(t_sub).as_ndarray()])
                t += trajectory.tf

            acc = np.array(acc)
            states = np.array([
                np.concatenate(
                    ([policy_path_ts[i]], self.policy_path[i].as_ndarray()))
                for i in range(len(self.policy_path))
            ])
            ax2.step(acc[:, 0], acc[:, 1], label="$a_l$")
            ax2.step(acc[:, 0], acc[:, 2], label="$a_r$")
            # ax2.plot(states[:, 0], states[:, 1], label="$p_x$")
            # ax2.plot(states[:, 0], states[:, 2], label="$p_y$")
            ax2.plot(states[:, 0], states[:, 3], label="$\psi$")
            ax2.plot(states[:, 0], states[:, 4], label="$v_x$")
            ax2.plot(states[:, 0], states[:, 5], label="$v_y$")
            ax2.plot(states[:, 0], states[:, 6], label="$d\psi$")
            ax2.legend()

            f.show()
            plt.show()

        return policy

    def test_dynamics(self, init: SpacecraftState, policy, total_time: float):
        sp = SpacecraftParameters.default()
        model = SpacecraftModel(init, self.sg, sp)
        dt = 0.05
        states = [init]
        t = 0
        ts = [0]
        while t < total_time:
            model.update(policy(t), dt)
            ts.append(t + dt)
            states.append(model._state)
            t += dt
        return ts, states

    def plan_rrt_path(self, spacecraft_state: SpacecraftState) -> List[Node]:

        # add start point to point-cloud (pc)
        root_idx = self._add_root(spacecraft_state)

        # build kdtree and determine distance of each point from Spacecraft
        self.distance.init_tree(self.sampler.pc2array())
        distance = self.distance.get_distance(
            root_idx, point_cloud=self.sampler.pc2array())
        distance_idx_sorted = np.argsort(distance)

        # initialize distances between samples in the pc and the obstacles in the collision class
        self.collision_checker.obstacle_distance(
            self.sampler.pc2array())

        for idx in distance_idx_sorted:
            self._update_graph(idx)

        if plot:
            ax = plt.gca()
            ax = self._draw_obstacles(ax)
            self._plotter(ax)

        rrt_path = self._get_optimal_rrt_path()
        return self._rrt_path_improvement(rrt_path)

    def plan_motion_path(self, start: SpacecraftState,
                         rrt_path: List[Node]) -> List[SpacecraftTrajectory]:
        rrt_line = LineString([node.pos for node in rrt_path])
        # A*
        state = start
        deviation_costs = defaultdict(lambda: np.inf)
        deviation_costs[state] = 0
        costs = defaultdict(lambda: np.inf)
        costs[state] = 0

        parents = dict()
        frontier = []
        primitives = dict()
        primitives[state] = None
        heapq.heappush(frontier,
                       (costs[state] + deviation_costs[state], state))

        def is_goal(state: SpacecraftState) -> bool:
            in_goal = self.goal.goal.contains(Point([state.x, state.y]))
            speed = np.linalg.norm([state.vx, state.vy])
            slow = speed < MAX_GOAL_VEL

            if in_goal and not slow:
                print(f"too fast: {speed}")
            return in_goal and slow

        def cost(primitive: SpacecraftTrajectory) -> Tuple[float, float]:
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
            vel = np.array(
                [np.abs([state.vx, state.vy]) for state in primitive.states])
            control_input = np.array([
                primitive.commands[0].acc_left, primitive.commands[0].acc_right
            ])
            goal_dist = np.linalg.norm(rrt_path[-1].pos -
                                       np.array([state.x, state.y]))
            deviation_err = np.max(norms)
            control_err = np.linalg.norm(control_input)
            end_vel = np.linalg.norm(vel[-1, :])

            # collision cost
            # primitive_line = LineString(primitive_pos)
            # even_division = [
            #     primitive_line.interpolate(offset)
            #     for offset in np.arange(0, primitive_line.length, 1)
            # ]
            # even_pts = np.array([pt.xy for pt in even_division])
            is_collding, min_dist = self.collision_checker.collding(
                primitive_pos)
            collsion_cost = 0
            if is_collding:
                collsion_cost = np.inf
            else:
                collsion_cost = 1 / (min_dist - self.sg.w_half)

            # angle = primitive.states[-1].p
            # cost_err = primitive.get_cost()
            # print(deviation_err, control_err)
            return deviation_err**2, (end_vel / goal_dist) + collsion_cost

        def heuristic(state: SpacecraftState) -> float:
            # path_dist = rrt_line.length - rrt_line.project(Point([state.x, state.y]))
            # if path_dist < 5:
            pos = np.array([state.x, state.y])
            goal = rrt_path[-1].pos
            dist = np.linalg.norm(goal - pos)
            return dist

        i = 0
        while len(frontier) > 0:
            prio, state = heapq.heappop(frontier)
            i += 1
            if i % 10 == 0:
                print(
                    f"{i}: priority: {prio:.2f}, cost: {deviation_costs[state]:.2f} + {costs[state]:.2f} = {deviation_costs[state] + costs[state]:.2f}, heuristic: {heuristic(state):.2f}"
                )
            if is_goal(state):
                print(
                    f"{i}: priority: {prio:.2f}, cost: {deviation_costs[state]:.2f} + {costs[state]:.2f} = {deviation_costs[state] + costs[state]:.2f}, heuristic: {heuristic(state):.2f}"
                )
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
                return motion_path[1:], primitives
            for primitive in self.motion_primitives.get_primitives_from(state):
                end_state = primitive.states[-1]

                deviation_cost, c = cost(primitive)
                deviation_cost = max(deviation_costs[state], deviation_cost)
                c = costs[state] + c
                total_cost = deviation_cost + c

                primitives[end_state] = primitive
                f = total_cost + heuristic(end_state)
                if total_cost < deviation_costs[end_state]:
                    deviation_costs[end_state] = deviation_cost
                    costs[end_state] = c
                    parents[end_state] = state
                    heapq.heappush(frontier, (f, end_state))
        raise RuntimeError("could not find motion path from start to goal")

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

        # init x_min and its cost
        x_min = None
        c_min = np.inf  # x_new.cost

        # check for all samples within radius which results in the smallest cost to reach the new sample
        near_nodes = [self.tree_idx[idx] for idx in X_near_idx_prune]
        for idx, x_near in enumerate(near_nodes):
            pt_distance = self.cost_fct(x_near.pos, x_rand)
            # currently euclidean distance, if changed, cannot be passed anymore to collision check
            collision_free = self.collision_checker.path_collision_free(
                x_near.pos, x_rand, pt_distance, idx)
            if collision_free and x_near.cost + pt_distance < c_min:
                x_min = x_near
                c_min = x_near.cost + pt_distance

        # if no feasible path could be found, terminate update
        if not x_min:
            return

        goal_state = SpacecraftState(x=x_rand[0],
                                     y=x_rand[1],
                                     psi=0,
                                     vx=0,
                                     vy=0,
                                     dpsi=0)

        # add new node to tree
        x_new = Node(state=goal_state, cost=c_min)
        self.tree.add_node(x_new)
        self.tree_idx[x_idx] = x_new
        self.tree.add_edge(x_min,
                           x_new,
                           trajectory=SpacecraftTrajectory(
                               [], [x_min.state, x_new.state], 0, 0, 0))

        # rebuild tree s.t. samples that can be reached with a smaller cost from the x_new are updated
        for x_near in near_nodes:
            pt_distance = self.cost_fct(x_rand, x_near.pos)
            # currently euclidean distance, if changed, cannot be passed anymore to collision check
            collision_free = self.collision_checker.path_collision_free(
                x_rand, x_near.pos, pt_distance, x_idx)
            motion_cost = c_min + pt_distance  # motion_cost
            if c_min + motion_cost < x_near.cost and collision_free:
                x_parent = self.tree.predecessors(x_near)
                self.tree.remove_edge(next(x_parent), x_near)
                self.tree.add_edge(x_new,
                                   x_near,
                                   trajectory=SpacecraftTrajectory(
                                       [], [x_new.state, x_near.state], 0, 0,
                                       0))

    def _get_optimal_rrt_path(self) -> List[Node]:
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

    def _rrt_path_improvement(self, rrt_path: List[Node]):
        # remove point of the path if curvature is smaller than a threshold
        for i in range(PRUNE_ITERATIONS):
            path_mask = [True]
            for idx in range(1, len(rrt_path) - 1, 1):
                coeff = np.polyfit(x=[
                    rrt_path[idx - 1].state.x, rrt_path[idx].state.x,
                    rrt_path[idx + 1].state.x
                ],
                                   y=[
                                       rrt_path[idx - 1].state.y,
                                       rrt_path[idx].state.y,
                                       rrt_path[idx + 1].state.y
                                   ],
                                   deg=2)
                path_mask.append(
                    True
                ) if coeff[0] > MIN_CURVATURE / 2 else path_mask.append(False)
            path_mask.append(True)
            rrt_path = [
                node for idx, node in enumerate(rrt_path) if path_mask[idx]
            ]

        return rrt_path

    def _draw_obstacles(self, ax):
        shapely_viz = ShapelyViz(ax)
        # plot static obstacles
        for s_obstacle in self.static_obstacles:
            shapely_viz.add_shape(s_obstacle.shape,
                                  color=s_obstacle.geometry.color,
                                  zorder=ZOrders.ENV_OBSTACLE)

        # comptue dynamic occupancies and plot them
        for d_obs in self.dynamic_simulator.compute_occupancies().values():
            shapely_viz.add_shape(d_obs,
                                  color="green",
                                  zorder=ZOrders.ENV_OBSTACLE)

        # plot goal
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
