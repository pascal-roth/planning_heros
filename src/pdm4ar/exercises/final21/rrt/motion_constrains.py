from typing import Tuple

from dg_commons.sim.models.model_structures import ModelParameters
from dg_commons.sim.models.spacecraft_structures import SpacecraftParameters

from pdm4ar.exercises.final21.rrt.params import *
from pdm4ar.exercises import logger


class MotionConstrains:
    def __init__(self):
        self.limit_acc: Tuple[float, float] = (-CONSTRAIN_ACC, CONSTRAIN_ACC)
        self.limit_vel: Tuple[float,
                              float] = (-self._kmh2ms(CONSTRAIN_VEL_LIN),
                                        self._kmh2ms(CONSTRAIN_VEL_LIN))
        self.limit_dpsi: Tuple[float,
                               float] = (-CONSTRAIN_VEL_ANG, CONSTRAIN_VEL_ANG)

    def spacecraft_params(self) -> SpacecraftParameters:
        return SpacecraftParameters(self.limit_vel, self.limit_acc,
                                    self.limit_dpsi)

    def check_vel_limit(self, vel: float) -> None:
        if vel <= self.limit_vel[0] or vel >= self.limit_vel[1]:
            logger.warn(
                f"Reached min or max velocity: {vel:.2f}  with speed limits "
                f"[{self.limit_vel[0]:.2f},{self.limit_vel[1]:.2f}]")

    def check_acc_limit(self, acc: float) -> None:
        if acc <= self.limit_acc[0] or acc >= self.limit_acc[1]:
            logger.warn(
                f"Reached min or max acceleration: {acc:.2f} with acceleration limits "
                f"[{self.limit_acc[0]:.2f},{self.limit_acc[1]:.2f}]")

    def check_dpsi_limit(self, psi: float) -> None:
        if psi <= self.limit_dpsi[0] or psi >= self.limit_dpsi[1]:
            logger.warn(f"Reached min or max psi: {psi:.2f}  with psi limits "
                        f"[{self.limit_dpsi[0]:.2f},{self.limit_dpsi[1]:.2f}]")

    def apply_vel_limit(self, vel: float) -> float:
        return max(self.limit_vel[0], min(vel, self.limit_vel[1]))

    def apply_acc_limit(self, acc: float) -> float:
        return max(self.limit_acc[0], min(acc, self.limit_acc[1]))

    def apply_dpsi_limit(self, psi: float) -> float:
        return max(self.limit_dpsi[0], min(psi, self.limit_dpsi[1]))

    @staticmethod
    def _kmh2ms(vel: float) -> float:
        return vel / 3.6
