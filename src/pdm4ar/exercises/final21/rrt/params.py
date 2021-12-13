import numpy as np
# parameters used over different files

CONSTRAIN_ACC: int = 10
CONSTRAIN_VEL_LIN: int = 50
CONSTRAIN_VEL_ANG: float = 2 * np.pi  #2 * np.pi

# Motion primitives
DELTAT_LIMITS = (.5, 2.)
MOTION_PRIMITIVE_INPUT_DIVISIONS: int = 10
STEERING_MAX_DIST: float = 10.
MAX_ABS_ACC_DIFF: float = 2.
