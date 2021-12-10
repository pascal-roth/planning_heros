from typing import List
from dg_commons.sim.models.spacecraft import SpacecraftCommands, SpacecraftState

class SpacecraftTrajectory:
    def __init__(self, commands: List[SpacecraftCommands],
                 states: List[SpacecraftState]):
        self.commands = commands
        self.states = states