from typing import Tuple

from pdm4ar.exercises.final21.rrt.params import *
from pdm4ar.exercises import logger


class MotionConstrains:
    def __init__(self):
        self.limit_acc: Tuple[float, float] = (-CONSTRAIN_ACC, CONSTRAIN_ACC)
        self.limit_vel: Tuple[float, float] = (- self._kmh2ms(CONSTRAIN_VEL_LIN), self._kmh2ms(CONSTRAIN_VEL_LIN))
        self.limit_psi: Tuple[float, float] = (-CONSTRAIN_VEL_ANG, CONSTRAIN_VEL_ANG)

    def check_vel_limit(self, vel: float) -> None:
        if vel <= self.limit_vel[0] or vel >= self.limit_vel[1]:
            logger.warn(f"Reached min or max velocity: {vel:.2f}  with speed limits "
                        f"[{self.limit_vel[0]:.2f},{self.limit_vel[1]:.2f}]")

    def check_acc_limit(self, acc: float) -> None:
        if acc <= self.limit_acc[0] or acc >= self.limit_acc[1]:
            logger.warn(f"Reached min or max acceleration: {acc:.2f} with acceleration limits "
                        f"[{self.limit_acc[0]:.2f},{self.limit_acc[1]:.2f}]")

    def check_psi_limit(self, psi: float) -> None:
        if psi <= self.limit_psi[0] or psi >= self.limit_psi[1]:
            logger.warn(f"Reached min or max psi: {psi:.2f}  with psi limits "
                        f"[{self.limit_psi[0]:.2f},{self.limit_psi[1]:.2f}]")

    @staticmethod
    def _kmh2ms(vel: float) -> float:
        return vel / 3.6
