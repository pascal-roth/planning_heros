import math
from scipy.integrate import solve_ivp
from dg_commons.sim.models.spacecraft import SpacecraftState, SpacecraftCommands, SpacecraftGeometry

from pdm4ar.exercises.final21.rrt.motion_constrains import MotionConstrains


class MPrimitives:
    def __init__(self, spacecraft_geometry: SpacecraftGeometry, time_step: float = 0.1):
        self.sg: SpacecraftGeometry = spacecraft_geometry
        self.dt: float = time_step

        self.motion_constrains: MotionConstrains = MotionConstrains()


    def get_final_spacecraft_state(self, spacecraft_t0: SpacecraftState, acc_left: float, acc_right: float):
        acc_x = (acc_left + acc_right) * math.sin(spacecraft_t0.psi)
        acc_y = (acc_left + acc_right) * math.sin(spacecraft_t0.psi)

        solve_ivp(fun=self._dynamics, t_span=(0.0, float(self.dt)), y0=spacecraft_t0)


    def _dynamics(self, x0: SpacecraftState, u: SpacecraftCommands) -> SpacecraftState:
        self.motion_constrains.check_acc_limit(u.acc_left)
        self.motion_constrains.check_acc_limit(u.acc_right)

        acc_sum = u.acc_right + u.acc_left
        acc_diff = u.acc_right - u.acc_left

        vx = x0.vx
        vy = x0.vy
        costh = math.cos(x0.psi)
        sinth = math.sin(x0.psi)
        dx = vx * costh - vy * sinth
        dy = vx * sinth + vy * costh

        ax = acc_sum + x0.vy * x0.dpsi
        ay = -x0.vx * x0.dpsi
        ddpsi = self.sg.w_half * self.sg.m / self.sg.Iz * acc_diff  # need to be saturated first
        return SpacecraftState(x=dx, y=dy, psi=x0.dpsi, vx=ax, vy=ay, dpsi=ddpsi)



