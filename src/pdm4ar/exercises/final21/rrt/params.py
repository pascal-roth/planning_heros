import numpy as np
# parameters used over different files

CONSTRAIN_ACC: int = 10
CONSTRAIN_VEL_LIN: int = 50
CONSTRAIN_VEL_ANG: float = 2 * np.pi

# Motion primitives
DELTAT_LIMIT = 1
MOTION_PRIMITIVE_INPUT_DIVISIONS: int = 20
MOTION_PRIMITIVE_STATE_DIVISIONS: int = 5
STEERING_MAX_DIST: float = 10.
MAX_ABS_ACC_DIFF: float = 2

MAX_GOAL_VEL: float = 10000

# Prune RRT path
MIN_CURVATURE = 1e-3
PRUNE_ITERATIONS = 1

# Buffer (Inflate) Obstacles by Distance
BUFFER_DISTANCE = 4
