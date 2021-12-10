import numpy as np


def euclidean_cost(x_start: np.ndarray, x_end: np.ndarray) -> float:
    return np.linalg.norm(x_end - x_start)
