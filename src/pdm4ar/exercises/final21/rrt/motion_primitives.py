from typing import List, Tuple, Dict
import numpy as np
from scipy.integrate import solve_ivp
from dg_commons.sim.models.spacecraft import SpacecraftCommands, SpacecraftState, SpacecraftGeometry

import matplotlib.pyplot as plt
import matplotlib

from pdm4ar.exercises.final21.rrt.motion_constrains import MotionConstrains
from pdm4ar.exercises.final21.rrt.params import CONSTRAIN_VEL_ANG, DELTAT_LIMITS
from scipy.spatial import KDTree


class SpacecraftTrajectory:
    def __init__(self, commands: List[SpacecraftCommands],
                 states: List[SpacecraftState], t0: float, tf: float,
                 dt: float):
        self.commands = commands
        self.states = states
        assert t0 <= tf, f"SpacecraftTrajectory start time needs to be less than it's end time. Was: {tf} > {t0}"
        self.t0 = t0
        self.tf = tf
        self.dt = dt

    def get_cost(self) -> float:
        return float(self.get_dist() / (self.tf - self.t0))

    def get_dist(self) -> float:
        dist = 0.
        for i in range(1, len(self.states)):
            p = self.states[i - 1]
            c = self.states[i]
            p_pos = np.array([p.x, p.y])
            c_pos = np.array([c.x, c.y])
            dist += np.linalg.norm(p_pos - c_pos)
        return dist


    def offset(self, x: float, y: float) -> 'SpacecraftTrajectory':
        return SpacecraftTrajectory(self.commands.copy(), [
            SpacecraftState(state.x + x, state.y + y, state.psi, state.vx,
                            state.vy, state.dpsi) for state in self.states
        ], self.t0, self.tf, self.dt)


class TrajectoryGroup:
    def __init__(self, trajectories: List[SpacecraftTrajectory]):
        self.trajectories = trajectories
        self.final_pos = np.array([[t.states[-1].x, t.states[-1].y]
                                   for t in self.trajectories])
        self.kdtree = KDTree(self.final_pos)


class MotionPrimitives:
    def __init__(self,
                 spacecraft_geometry: SpacecraftGeometry,
                 steps: int = 10):
        self.sg: SpacecraftGeometry = spacecraft_geometry
        self.steps = steps

        self.motion_constrains: MotionConstrains = MotionConstrains()
        self.primitives: List[SpacecraftTrajectory] = []
        self.max_distance_covered = self._max_distance_covered()

        self.primitive_database: Dict[str, TrajectoryGroup] = dict()

    def distance(self, start: SpacecraftState,
                 end: SpacecraftState) -> Tuple[float, SpacecraftTrajectory]:
        discrete = self._discretize_state(start)

        # get trajectory group and create it if not present
        discrete_repr = repr(discrete)
        if discrete_repr not in self.primitive_database:
            print(f"building new TrajectoryGroup for {discrete}")
            self.primitive_database[discrete_repr] = TrajectoryGroup(
                self._generate_trajectories(discrete))
        traj_group = self.primitive_database[discrete_repr]

        # find closest end state
        _, closest_idx = traj_group.kdtree.query(
            [end.x - start.x, end.y - start.y])
        trajectory: SpacecraftTrajectory = traj_group.trajectories[closest_idx]

        trajectory = trajectory.offset(start.x, start.y)
        return trajectory.get_cost(), trajectory

    def _discretize_state(self, state: SpacecraftState) -> SpacecraftState:
        def closest(value: float, steps: np.ndarray) -> float:
            norm = (steps - value)**2
            c_idx = np.argmin(norm)
            return steps[c_idx]

        limit_vel = self.motion_constrains.limit_vel
        limit_dpsi = self.motion_constrains.limit_dpsi
        steps = 5
        return SpacecraftState(
            x=0,
            y=0,
            psi=closest(state.psi,
                        np.linspace(0, 2 * np.pi, steps)[:-1]),
            vx=closest(state.vx, np.linspace(limit_vel[0], limit_vel[1],
                                             steps)),
            vy=closest(state.vy, np.linspace(limit_vel[0], limit_vel[1],
                                             steps)),
            dpsi=closest(state.dpsi,
                         np.linspace(limit_dpsi[0], limit_dpsi[1], steps)))

    def _generate_trajectories(
            self, state: SpacecraftState) -> List[SpacecraftTrajectory]:
        input_limits = self.motion_constrains.limit_acc
        acc = np.linspace(input_limits[0], input_limits[1], self.steps)
        dts = np.linspace(DELTAT_LIMITS[0], DELTAT_LIMITS[1], self.steps)
        acc_left, acc_right = np.meshgrid(acc, acc)
        primitives = []

        for k in range(dts.shape[0]):
            dt = dts[k]
            for i in range(acc_left.shape[0]):
                for j in range(acc_left.shape[1]):
                    command = SpacecraftCommands(acc_left[i, j], acc_right[i,
                                                                           j])
                    trajectory = self._get_trajectory(state, command, dt)
                    primitives.append(trajectory)
        return primitives

    def _get_trajectory(self,
                        spacecraft_t0: SpacecraftState,
                        u: SpacecraftCommands,
                        tf: float,
                        dt: float = 0.1) -> SpacecraftTrajectory:
        # express initial state
        y0 = np.array([
            spacecraft_t0.x, spacecraft_t0.y, spacecraft_t0.vx,
            spacecraft_t0.vy, spacecraft_t0.psi, spacecraft_t0.dpsi
        ])

        def dynamics(t, y):
            px, py, vx, vy, psi, dpsi = y
            # only apply acceleration in the beginning
            # if t <= dt:
            acc_left = self.motion_constrains.apply_acc_limit(u.acc_left)
            acc_right = self.motion_constrains.apply_acc_limit(u.acc_right)
            # else:
            #     acc_left = 0
            #     acc_right = 0

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

        sol = solve_ivp(fun=dynamics, t_span=(0.0, tf), y0=y0, vectorized=True)

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
        return SpacecraftTrajectory([u], states, 0., tf, dt)

    def _max_distance_covered(self) -> float:
        fastest_state = SpacecraftState(x=0,
                                        y=0,
                                        psi=np.deg2rad(45),
                                        vx=self.motion_constrains.limit_vel[1],
                                        vy=self.motion_constrains.limit_vel[1],
                                        dpsi=0)
        primitives = self._generate_trajectories(fastest_state)
        return np.max([trajectory.get_cost() for trajectory in primitives])

    # def plot_primitives(self):
    #     plt.figure()
    #     for trajectory in self.primitives:
    #         x = [s.x for s in trajectory.states]
    #         y = [s.y for s in trajectory.states]
    #         plt.plot(x, y)
    #     plt.show()


if __name__ == "__main__":
    matplotlib.use("TkAgg")
    sg = SpacecraftGeometry.default()
    mp = MotionPrimitives(sg)
    spacecraft_t0 = SpacecraftState(x=0, y=0, psi=0, vx=.5, vy=0, dpsi=0)
    mp.distance(spacecraft_t0)
    mp.plot_primitives()
