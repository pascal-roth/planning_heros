import numpy as np
# parameters used over different files

CONSTRAIN_ACC: int = 10
CONSTRAIN_VEL_LIN: int = 50
CONSTRAIN_VEL_ANG: float = 2 * np.pi

# Motion primitives
DELTAT = 1
MOTION_PRIMITIVE_INPUT_DIVISIONS: int = 20
MOTION_PRIMITIVE_STATE_DIVISIONS: int = 5
STEERING_MAX_DIST: float = 10.
MAX_ABS_ACC_DIFF: float = 1.5

MAX_GOAL_VEL: float = 100

# Prune RRT path
MIN_CURVATURE = 1e-3
PRUNE_ITERATIONS = 1

# Buffer (Inflate) Obstacles by Distance
STATIC_BUFFER_DISTANCE = 2
DYNAMIC_BUFFER_DISTANCE = 1

# Game world size 0 -> 100 in x,y
WORLD_SIZE = (100, 100)


#
# Planning for dynamic obstacles
#

# Time in seconds to plan ahead for
PLANNING_HORIZON:float = 1.
