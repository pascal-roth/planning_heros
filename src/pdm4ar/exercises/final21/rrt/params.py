import numpy as np
# parameters used over different files

CONSTRAIN_ACC: int = 10
CONSTRAIN_VEL_LIN: int = 50
CONSTRAIN_VEL_ANG: float = 2 * np.pi

# Motion primitives
DELTAT_LIMITS = (.75, 2.)
MOTION_PRIMITIVE_INPUT_DIVISIONS: int = 5
STEERING_MAX_DIST: float = 10.
