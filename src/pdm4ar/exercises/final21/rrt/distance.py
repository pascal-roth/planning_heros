import numpy as np
import enum
from typing import Optional, Callable, Union, Tuple
from scipy.spatial import KDTree


class DistanceMetric(enum.Enum):
    # Euclidean Distance, i.e. L2 norm
    L2 = 0


class Distance:
    def __init__(self, metric: DistanceMetric = DistanceMetric.L2):
        """
        :param metric
        """
        self.metric: DistanceMetric = metric

        self.metric_fct: Callable

        if self.metric == DistanceMetric.L2:
            self.metric_fct = self._euclidean_distance
        else:
            raise NotImplementedError(
                f'Distance Metric {self.metric} not implemented')

        self.tree: KDTree

    def init_tree(self, samples: np.ndarray) -> None:
        self.tree = KDTree(samples)

    def get_distance(self, point_idx: int, point_cloud: np.ndarray):
        distance = [
            self.metric_fct(point_cloud[point_idx] - pc_entry)
            for pc_entry in point_cloud
        ]
        # the point we compare the distance with is already included in the pointcloud -> set its distance to inf
        distance[point_idx] = np.inf
        return distance

    def dist(self, x: np.ndarray, y: np.ndarray) -> float:
        return self.metric_fct(x - y)

    @staticmethod
    def _euclidean_distance(x: np.ndarray):
        return np.linalg.norm(x)
