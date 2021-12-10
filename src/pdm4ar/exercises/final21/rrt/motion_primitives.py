from typing import List
import numpy as np
from scipy.integrate import solve_ivp
from dg_commons.sim.models.spacecraft import SpacecraftCommands, SpacecraftState, SpacecraftGeometry

import matplotlib.pyplot as plt
import matplotlib

from pdm4ar.exercises.final21.rrt.motion_constrains import MotionConstrains
from pdm4ar.exercises.final21.rrt.trajectory import SpacecraftTrajectory


class MotionPrimitives:
    def __init__(self,
                 spacecraft_geometry: SpacecraftGeometry,
                 time_step: float = 0.1):
        self.sg: SpacecraftGeometry = spacecraft_geometry
        self.dt: float = time_step

        self.motion_constrains: MotionConstrains = MotionConstrains()
        self.primitives: List[SpacecraftTrajectory] = []

    def generate(self,
                 spacecraft_t0: SpacecraftState,
                 steps: int = 10) -> List[SpacecraftTrajectory]:
        """Generates all motion primitives for a given initial state

        Args:
            spacecraft_t0 (SpacecraftState): initial state for the primitives
            steps (int, optional): Number of divisions for input accelerations. Defaults to 10.

        Returns:
            List[SpacecraftTrajectory]: List of all generated trajectories
        """
        input_limits = self.motion_constrains.limit_acc
        acc = np.linspace(input_limits[0], input_limits[1], steps)
        acc_left, acc_right = np.meshgrid(acc, acc)

        for i in range(acc_left.shape[0]):
            for j in range(acc_left.shape[1]):
                command = SpacecraftCommands(acc_left[i, j], acc_right[i, j])
                trajectory = self._get_trajectory(spacecraft_t0, command)
                self.primitives.append(trajectory)
        return self.primitives

    def _get_trajectory(self, spacecraft_t0: SpacecraftState,
                        u: SpacecraftCommands) -> SpacecraftTrajectory:
        # express initial state
        y0 = np.array([
            spacecraft_t0.x, spacecraft_t0.y, spacecraft_t0.vx,
            spacecraft_t0.vy, spacecraft_t0.psi, spacecraft_t0.dpsi
        ])

        def dynamics(t, y):
            px, py, vx, vy, psi, dpsi = y
            acc_left = self.motion_constrains.apply_acc_limit(u.acc_left)
            acc_right = self.motion_constrains.apply_acc_limit(u.acc_right)

            acc_sum = acc_right + acc_left
            acc_diff = acc_right - acc_left

            costh = np.cos(psi)
            sinth = np.sin(psi)
            dx = vx * costh - vy * sinth
            dy = vx * sinth + vy * costh

            dvx = acc_sum + vy * dpsi
            dvy = -vx * dpsi
            dpsi = psi
            ddpsi = self.sg.w_half * self.sg.m / self.sg.Iz * acc_diff  # need to be saturated first
            ddpsi = self.motion_constrains.apply_dpsi_limit(ddpsi)

            ret = np.zeros((6, ))
            ret[0] = px + dx
            ret[1] = py + dy
            ret[2] = dvx
            ret[3] = dvy
            ret[4] = dpsi
            ret[5] = ddpsi
            return ret

        sol = solve_ivp(fun=dynamics,
                        t_span=(0.0, self.dt),
                        y0=y0,
                        vectorized=True)
        assert sol.success, f"Solving the IVP for ({u.acc_left}, {u.acc_right}) failed"
        states: List[SpacecraftState] = [None] * sol.y.shape[1]
        for i in range(sol.y.shape[1]):
            s = sol.y[:, i]
            states[i] = SpacecraftState(x=s[0],
                                        y=s[1],
                                        psi=s[4],
                                        vx=s[2],
                                        vy=s[3],
                                        dpsi=s[5])
        return SpacecraftTrajectory([u], states)

    def plot_primitives(self):
        plt.figure()
        for trajectory in self.primitives:
            x = [s.x for s in trajectory.states]
            y = [s.y for s in trajectory.states]
            plt.plot(x, y)
        plt.show()


if __name__ == "__main__":
    matplotlib.use("TkAgg")
    sg = SpacecraftGeometry.default()
    mp = MotionPrimitives(sg)
    spacecraft_t0 = SpacecraftState(x=0, y=0, psi=0, vx=.5, vy=0, dpsi=0)
    mp.generate(spacecraft_t0)
    mp.plot_primitives()
