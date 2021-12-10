import numpy as np
import enum
from typing import Optional, Callable, Union, Tuple


class DistanceMethod(enum.Enum):
    # brute force distance calculation
    BF = 0,
    # kdtree
    KDTREE = 1,


class DistanceMetric(enum.Enum):
    # Euclidean Distance, i.e. L2 norm
    L2 = 0


class Distance:
    def __init__(self,
                 method: DistanceMethod = DistanceMethod.BF,
                 metric: DistanceMetric = DistanceMetric.L2):
        """
        :param metric
        :param method:
        """
        self.method: DistanceMethod = method
        self.metric: DistanceMetric = metric

        self.metric_fct: Callable
        self._get_metric()

    def get_nn(self, point_idx: int, point_cloud: np.ndarray, radius: Optional[float] = None) -> \
            Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        distance = self._get_distance(point_idx, point_cloud)
        # if radius is defined, give all samples within radius, otherwise just the clostest idx of the pointcloud
        if radius:
            distance_mask = [True if distance_current < radius else False for distance_current in distance]
            distance_idx = np.arange(len(distance))
            return np.argmin(distance), distance_idx[distance_mask]
        else:
            return np.argmin(distance)

    def _get_distance(self, point_idx: int, point_cloud: np.ndarray, ):
        if self.method == DistanceMethod.BF:
            distance = [self.metric_fct(point_cloud[point_idx] - pc_entry) for pc_entry in point_cloud]
        else:
            raise NotImplementedError(f'Distance Method {self.method} not implemented')

        # the point we compare the distance with is already included in the pointcloud -> set its distance to inf
        distance[point_idx] = np.inf
        return distance

    def _get_metric(self):
        if self.metric == DistanceMetric.L2:
            self.metric_fct = self._euclidean_distance
        else:
            raise NotImplementedError(f'Distance Metric {self.metric} not implemented')

    @staticmethod
    def _euclidean_distance(x: np.ndarray):
        return np.linalg.norm(x)

