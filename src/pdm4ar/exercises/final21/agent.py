from typing import Sequence

from dg_commons import PlayerName
from dg_commons.planning import PolygonGoal
from dg_commons.sim import SimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.spacecraft import SpacecraftCommands
from dg_commons.sim.models.spacecraft_structures import SpacecraftGeometry

from pdm4ar.exercises.final21.rrt.rrt import RRT
from pdm4ar.exercises.final21.rrt.params import DEFAULT_DT


class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do NOT modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""

    def __init__(self,
                 goal: PolygonGoal,
                 static_obstacles: Sequence[StaticObstacle],
                 sg: SpacecraftGeometry,
                 sp: SpacecraftGeometry):
        self.goal = goal
        self.static_obstacles = static_obstacles
        self.sg = sg
        self.sp = sp
        self.name = None

        self.rrt: RRT = None
        self.prev_sim_time:float = -1.

    def on_episode_init(self, my_name: PlayerName):
        self.name = my_name

    def get_commands(self, sim_obs: SimObservations) -> SpacecraftCommands:
        """ This method is called by the simulator at each time step.

        This is how you can get your current state from the observations:
        my_current_state: SpacecraftState = sim_obs.players[self.name].state

        :param sim_obs:
        :return:
        """
        # estimate dt from simulation time
        # TODO: update DEFAULT_DT such that dt stays constant
        dt = sim_obs.time - self.prev_sim_time if self.prev_sim_time != -1. else DEFAULT_DT
        self.prev_sim_time = sim_obs.time
        if self.rrt is None:
            print(f"Creating RRT with dt: {dt}s")
            self.rrt = RRT(self.goal, self.static_obstacles, self.sg, dt=dt)

        self.rrt.plan_path(sim_obs.players['PDM4AR'].state)

        return SpacecraftCommands(acc_left=1, acc_right=1)
