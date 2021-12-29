import numpy as np
from dg_commons.sim.models.spacecraft import SpacecraftState


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