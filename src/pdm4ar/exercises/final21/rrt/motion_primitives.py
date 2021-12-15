from typing import List, Tuple, Dict, Callable
from dg_commons.sim.models.model_utils import apply_acceleration_limits, apply_rot_speed_constraint
from dg_commons.sim.models.spacecraft_structures import SpacecraftParameters
import numpy as np
import math
from numpy.matrixlib import defmatrix
from scipy.integrate import solve_ivp
from dg_commons.sim.models.spacecraft import SpacecraftCommands, SpacecraftModel, SpacecraftState, SpacecraftGeometry

import matplotlib.pyplot as plt
import matplotlib

from pdm4ar.exercises.final21.rrt.motion_constrains import MotionConstrains
from pdm4ar.exercises.final21.rrt.params import CONSTRAIN_VEL_ANG, DELTAT, MAX_ABS_ACC_DIFF, MOTION_PRIMITIVE_STATE_DIVISIONS
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

    def rotate(self, psi: float) -> 'SpacecraftTrajectory':
        # TODO: untested DO NOT USE AS IS
        pos = np.array([[s.x, s.y] for s in self.states])
        pos_zero = pos - pos[0, :]
        R = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
        rotated = (R @ pos_zero.T).T
        return SpacecraftTrajectory(self.commands.copy(), [
            SpacecraftState(s[0], s[1], self.states[i].psi - psi,
                            self.states[i].vx, self.states[i].vy,
                            self.states[i].dpsi) for i, s in enumerate(rotated)
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

    def get_primitives_from(self,
                            start: SpacecraftState,
                            plot=False) -> List[SpacecraftTrajectory]:
        # discrete = self._discretize_state(start)

        # get trajectory group and create it if not present
        # discrete_repr = repr(discrete)
        # if discrete_repr not in self.primitive_database:
        #     start_vel = np.array([start.vx, start.vy])
        #     end_vel = np.array([discrete.vx, discrete.vy])
        #     print(f"discretization velocity error: {np.linalg.norm(start_vel - end_vel)}")
        #     group = TrajectoryGroup(self._generate_trajectories(discrete))
        #     print(
        #         f"building new TrajectoryGroup of {len(group.trajectories)} primitives for {discrete}"
        #     )
        #     self.primitive_database[discrete_repr] = group
        #     if plot:
        #         # plot primitives
        #         plt.figure()
        #         matplotlib.use("TkAgg")
        #         plt.scatter(discrete.x,discrete.y, label="discrete start pos")
        #         plt.scatter(0, np.linalg.norm([start.vx, start.vy]), label="start vel")
        #         plt.scatter(0, np.linalg.norm([discrete.vx,discrete.vy]), label="discrete start vel")
        #         for trajectory in group.trajectories:
        #             pos = np.array([[s.x, s.y]
        #                             for s in trajectory.states])
        #             vel = np.array([np.linalg.norm([s.vx, s.vy]) for s in trajectory.states])
        #             vel_x = np.linspace(0, 1, vel.shape[0])
        #             plt.plot(pos[:, 0], pos[:, 1], color="orange")
        #             plt.plot(vel_x, vel, color="green")

        #         plt.legend()
        #         plt.show()

        return self._generate_trajectories(start)

    def _discretize_state(self, state: SpacecraftState) -> SpacecraftState:
        def closest(value: float, steps: np.ndarray) -> float:
            norm = (steps - value)**2
            c_idx = np.argmin(norm)
            return steps[c_idx]

        limit_vel = self.motion_constrains.limit_vel
        limit_dpsi = self.motion_constrains.limit_dpsi
        steps = MOTION_PRIMITIVE_STATE_DIVISIONS
        return SpacecraftState(
            x=0,
            y=0,
            psi=closest(state.psi,
                        np.linspace(0, 2 * np.pi, 5 * steps)[:-1]),
            vx=closest(state.vx,
                       np.linspace(limit_vel[0], limit_vel[1], 2 * steps)),
            vy=closest(state.vy,
                       np.linspace(limit_vel[0], limit_vel[1], 2 * steps)),
            dpsi=closest(state.dpsi,
                         np.linspace(limit_dpsi[0], limit_dpsi[1], steps)))

    def _generate_trajectories(
            self, state: SpacecraftState) -> List[SpacecraftTrajectory]:
        input_limits = self.motion_constrains.limit_acc
        acc = np.linspace(input_limits[0], input_limits[1], self.steps)
        acc_left, acc_right = np.meshgrid(acc, acc)
        primitives = []

        for i in range(acc_left.shape[0]):
            for j in range(acc_left.shape[1]):
                acc_diff = np.abs(acc_left[i, j] - acc_right[i, j])
                if acc_diff > MAX_ABS_ACC_DIFF:
                    continue
                command = SpacecraftCommands(acc_left[i, j], acc_right[i, j])
                trajectory = self._get_trajectory(state, command, DELTAT)
                primitives.append(trajectory)
        return primitives

    def _get_trajectory(self,
                        spacecraft_t0: SpacecraftState,
                        u: SpacecraftCommands,
                        tf: float,
                        dt: float = 0.1) -> SpacecraftTrajectory:
        # express initial state
        model = SpacecraftModel(spacecraft_t0, self.sg, SpacecraftParameters.default())
        y0 = spacecraft_t0.as_ndarray()

        def _stateactions_from_array(y: np.ndarray):
            n_states = SpacecraftState.get_n_states()
            state = SpacecraftState.from_array(y[0:n_states])

            actions = SpacecraftCommands(
                acc_left=y[SpacecraftCommands.idx["acc_left"] + n_states],
                acc_right=y[SpacecraftCommands.idx["acc_right"] + n_states],
            )
            return state, actions

        def dynamics(t, y):
            s0, actions = _stateactions_from_array(y=y)
            dx = model.dynamics(x0=s0, u=actions)
            du = np.zeros([len(SpacecraftCommands.idx)])
            return np.concatenate([dx.as_ndarray(), du])

        state_np = spacecraft_t0.as_ndarray()
        action_np = u.as_ndarray()
        y0 = np.concatenate([state_np, action_np])

        sol = solve_ivp(fun=dynamics,
                        t_span=(0.0, tf),
                        y0=y0, vectorized=False, method="RK23")

        assert sol.success, f"Solving the IVP for ({u.acc_left}, {u.acc_right}) failed"
        states: List[SpacecraftState] = []
        for i in range(sol.y.shape[1]):
            s = sol.y[:, i]
            states.append(SpacecraftState.from_array(s[:-2]))
        return SpacecraftTrajectory([u], states, 0., tf, dt)

    def _max_distance_covered(self) -> float:
        fastest_state = SpacecraftState(x=0,
                                        y=0,
                                        psi=np.deg2rad(45),
                                        vx=self.motion_constrains.limit_vel[1],
                                        vy=self.motion_constrains.limit_vel[1],
                                        dpsi=0)
        primitives = self._generate_trajectories(fastest_state)
        return np.max([trajectory.get_dist() for trajectory in primitives])

    def plot_primitives(self, primitives):
        matplotlib.use("TkAgg")
        plt.figure()
        for trajectory in primitives:
            x = [s.x for s in trajectory.states]
            y = [s.y for s in trajectory.states]
            plt.plot(x, y)
        plt.show()

    def test_dynamics(self,
                      start_state: SpacecraftState,
                      policy: Callable[[float], SpacecraftCommands],
                      tf: float,
                      dt: float = 0.05):

        t = 0
        states: List[SpacecraftState] = [start_state]
        while t <= tf:
            start_state = states[-1]
            new_states = self._get_trajectory(start_state, policy(t), dt,
                                              dt).states
            states += new_states
            t += dt

        return states


if __name__ == "__main__":
    matplotlib.use("TkAgg")
    sg = SpacecraftGeometry.default()
    mp = MotionPrimitives(sg)
    spacecraft_t0 = SpacecraftState(x=0, y=0, psi=1, vx=.5, vy=0, dpsi=4)
    primitives = mp._generate_trajectories(spacecraft_t0)
    mp.plot_primitives(primitives)
