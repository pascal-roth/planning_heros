from typing import List, Dict, Optional
from dg_commons.sim.models.spacecraft import SpacecraftCommands, SpacecraftState
from dg_commons.sim.models.spacecraft_structures import SpacecraftGeometry
from dg_commons.sim.simulator_structures import PlayerObservations
import shapely
import numpy as np

from pdm4ar.exercises.final21.rrt.motion_primitives import MotionPrimitives, SpacecraftTrajectory
from pdm4ar.exercises.final21.rrt.params import MIN_PLANNING_HORIZON, SAFTY_FACTOR, MAX_PLANNING_HORIZON
from pdm4ar.exercises.final21.rrt.node import Node
from shapely import affinity

from shapely.geometry import Polygon


class DynamicObstacleSimulator:
    def __init__(
        self,
        sg: SpacecraftGeometry,
        other_players: Dict[str, PlayerObservations],
        horizon: float = MIN_PLANNING_HORIZON,
        dt: float = 0.05,
    ):
        self.other_players = other_players
        self.sg = sg
        self.dt = dt
        self.horizon = horizon
        self.motion_primitives = MotionPrimitives(sg)

    def simulate(self, horizon) -> Dict[str, SpacecraftTrajectory]:
        trajectories: Dict[str, SpacecraftTrajectory] = dict()
        for name, player in self.other_players.items():
            # dynamic obstacles are assumed to not be controlled
            u = SpacecraftCommands(0, 0)
            trajectory = self.motion_primitives.get_trajectory(
                player.state, u, horizon, self.dt)
            trajectories[name] = trajectory
        return trajectories

    def compute_occupancies(self, horizon: Optional[float] = None) -> Dict[str, Polygon]:
        occupancies = dict()
        trajectories = self.simulate(self.horizon) if horizon is None else self.simulate(horizon)
        for name, trajectory in trajectories.items():
            player = self.other_players[name]
            shape = player.occupancy
            occupancy = player.occupancy
            init = trajectory.states[0]
            # union the occupancy of all succeeding states
            for state in trajectory.states[1:]:
                s = affinity.rotate(shape,
                                    angle=state.psi - init.psi,
                                    use_radians=False,
                                    origin='center')
                s = affinity.translate(s,
                                       xoff=state.x - init.x,
                                       yoff=state.y - init.y)
                occupancy = occupancy.union(s)
            # occupancies are over approximated by the convex hull
            occupancies[name] = occupancy.convex_hull
        return occupancies

    def get_planning_horizon(self, rrt_path: List[Node], sc_state: SpacecraftState) -> float:
        """
        Idea: determine distance between obstacle and the point on the rrt path/ motion primitive path where a collision
        might happen for different times, choose time as planning horizon which is as large as possible without
        violating a closeness constrain (if no collision between rrt and obstacle path, set planning horizon until
        dynamic obstacle hits other obstacle if that does not happen as well, set it to infty -> static case)

        :return:
        """
        rrt_points = [rrt_pt.pos for rrt_pt in rrt_path]
        sc_trajectory = shapely.geometry.LineString(rrt_points)
        do_point = [np.array([st.x, st.y]) for st in self.simulate(horizon=10)['DObs1'].states]
        do_trajectory = shapely.geometry.LineString(do_point)

        intersection = sc_trajectory.intersection(do_trajectory)

        if hasattr(intersection, 'x'):
            dist_sc_intersection = np.linalg.norm(np.array([intersection.x, intersection.y]) -
                                                  np.array([self.other_players['DObs1'].state.x,
                                                            self.other_players['DObs1'].state.y]))

            time_to_collision = dist_sc_intersection / sc_state.vx * SAFTY_FACTOR
        else:
            time_to_collision = np.infty

        if time_to_collision < MIN_PLANNING_HORIZON:
            return MIN_PLANNING_HORIZON
        elif time_to_collision > MAX_PLANNING_HORIZON:
            return MAX_PLANNING_HORIZON
        else:
            return time_to_collision
