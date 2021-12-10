import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Sequence, Dict, List
from scipy.stats import qmc
from shapely.geometry import Point

from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.planning import PolygonGoal
from dg_commons.maps.shapely_viz import ShapelyViz
from dg_commons.sim.simulator_visualisation import ZOrders

# for comparability
np.random.seed(seed=31)


class Sampler:
    """
    Sampler from the given geometry a certain number of samples
    """
    def __init__(self, static_obstacles: Sequence[StaticObstacle],
                 n_samples: int = 1000):
        """
        :param static_obstacles:    Obstacles in the given environment, note that obstacle 0 is the boundary of the
                                    environment !!!
        :param n_samples:           number of samples in initial point_cloud
        """
        # set goal and boundary obstacle to control conditions
        self.obstacles: Sequence[StaticObstacle] = static_obstacles
        # Initialite the Halton sampler
        self.sampler_fct: qmc.Halton = qmc.Halton(d=2, seed=31)
        # track how much sampels drawn from sequence
        self.n_samples: int = n_samples
        # point_cloud in the given environment
        self.point_cloud_idx_latest: int = n_samples
        self.point_cloud: Dict[int, np.ndarray] = self._init_point_cloud()

    def draw_samples(self, n_samples: int = 1) -> List[int]:
        """
        draw additional n_samples from distributiuon
        :param n_samples:
        :return:
        """
        # _ = self.sampler_fct.fast_forward(self.n_samples)   # TODO: unsure if necessary
        self.n_samples += n_samples
        samples = self._get_sample(n_samples)
        samples_idx = [self.point_cloud_idx_latest+i for i in range(n_samples)]
        self.add_sample(samples)
        return samples_idx

    def add_sample(self, samples) -> None:
        n_samples = len(samples)
        samples_pc = [(self.point_cloud_idx_latest+i, samples[i]) for i in range(n_samples)]
        self.point_cloud_idx_latest += n_samples
        self.point_cloud.update(dict(samples_pc))

    def pc2array(self) -> np.ndarray:
        pc_list = list(self.point_cloud.values())
        return np.array(pc_list)

    def _init_point_cloud(self) -> Dict[int, np.ndarray]:
        samples = self._get_sample(self.n_samples)
        point_cloud = [(i, samples[i]) for i in range(self.point_cloud_idx_latest)]
        return dict(point_cloud)

    def _get_sample(self, n_samples: int = 1) -> np.ndarray:
        """
        Initialize n samples of the given environment using the Halton Sequence.
        """
        # draw n_samples
        samples = self.sampler_fct.random(n_samples)
        # scale samples
        samples_scaled = self._scale_samples(samples)
        # check if samples in obstacle
        samples_checked = self._check_obstacles(samples_scaled)
        return samples_checked

    def _scale_samples(self, samples) -> np.ndarray:
        # scale samples
        lower_bound = self.obstacles[0].shape.bounds[:2]
        upper_bound = self.obstacles[0].shape.bounds[2:]
        return qmc.scale(samples, l_bounds=lower_bound, u_bounds=upper_bound)

    def _check_obstacles(self, samples: np.ndarray) -> np.ndarray:
        # convert sample to shaply points
        points = [Point(samples[idx, :]) for idx in range(len(samples))]
        # check if points in obstacles
        point_mask = [any([self.obstacles[idx+1].shape.contains(pt) for idx in range(len(self.obstacles)-1)]) for pt in points]
        point_mask = [not elem for elem in point_mask]
        samples_free = samples[point_mask]
        # if any points in obstacles, get new points of sequence and advance the sequence
        n_missing = len(samples)-len(samples_free)
        if n_missing > 0:
            # _ = self.sampler_fct.fast_forward(self.n_samples)  # TODO: unsure if necessary
            self.n_samples += n_missing
            samples_free = np.concatenate((samples_free, self._get_sample(n_missing)), axis=0)
        return samples_free

    def plot_samples(self, goal: PolygonGoal):
        matplotlib.use('TkAgg')
        ax = plt.gca()
        shapely_viz = ShapelyViz(ax)

        for s_obstacle in self.obstacles:
            shapely_viz.add_shape(s_obstacle.shape, color=s_obstacle.geometry.color, zorder=ZOrders.ENV_OBSTACLE)
        shapely_viz.add_shape(goal.get_plottable_geometry(), color="orange", zorder=ZOrders.GOAL, alpha=0.5)
        ax = shapely_viz.ax
        pc_array = self.pc2array()
        ax.scatter(pc_array[:, 0], pc_array[:, 1])
        ax.autoscale()
        ax.set_facecolor('k')
        ax.set_aspect("equal")
        plt.show()
