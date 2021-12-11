from typing import List
from dg_commons.sim.models.spacecraft import SpacecraftCommands, SpacecraftState
import numpy as np


class SpacecraftTrajectory:
    def __init__(self, commands: List[SpacecraftCommands],
                 states: List[SpacecraftState], t0: float, tf: float):
        self.commands = commands
        self.states = states
        assert t0 <= tf, f"SpacecraftTrajectory start time needs to be less than it's end time. Was: {tf} > {t0}"
        self.t0 = t0
        self.tf = tf

    def get_cost(self) -> float:
        cost = 0.
        for i in range(1, len(self.states)):
            p = self.states[i - 1]
            c = self.states[i]
            p_pos = np.array([p.x, p.y])
            c_pos = np.array([c.x, c.y])
            cost += np.linalg.norm(p_pos - c_pos)
        return float(cost / (self.tf - self.t0))
