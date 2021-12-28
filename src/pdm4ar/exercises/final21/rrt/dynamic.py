from typing import List, Dict
from dg_commons.sim.models.spacecraft import SpacecraftCommands
from dg_commons.sim.models.spacecraft_structures import SpacecraftGeometry
from dg_commons.sim.simulator_structures import PlayerObservations
import shapely

from pdm4ar.exercises.final21.rrt.motion_primitives import MotionPrimitives, SpacecraftTrajectory
from pdm4ar.exercises.final21.rrt.params import MOTION_PRIMITIVE_INPUT_DIVISIONS, PLANNING_HORIZON
from shapely import affinity

from shapely.geometry import Polygon
import matplotlib.pyplot as plt


class DynamicObstacleSimulator:
    def __init__(
        self,
        sg: SpacecraftGeometry,
        other_players: Dict[str, PlayerObservations],
        horizon: float = PLANNING_HORIZON,
        dt: float = 0.05,
    ):
        self.other_players = other_players
        self.sg = sg
        self.dt = dt
        self.horizon = horizon
        self.motion_primitives = MotionPrimitives(sg)
        self.trajectories = self.simulate()

    def simulate(self) -> Dict[str, SpacecraftTrajectory]:
        trajectories: Dict[str, SpacecraftTrajectory] = dict()
        for name, player in self.other_players.items():
            # dynamic obstacles are assumed to not be controlled
            u = SpacecraftCommands(0, 0)
            trajectory = self.motion_primitives.get_trajectory(
                player.state, u, self.horizon, self.dt)
            trajectories[name] = trajectory
        return trajectories

    def compute_occupancies(self) -> Dict[str, Polygon]:
        occupancies = dict()
        for name, trajectory in self.trajectories.items():
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
