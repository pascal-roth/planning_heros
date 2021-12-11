import numpy as np
# parameters used over different files

CONSTRAIN_ACC: int = 10
CONSTRAIN_VEL_LIN: int = 50
CONSTRAIN_VEL_ANG: float = 2 * np.pi

# Motion primitives
DEFAULT_DT: float = 0.1
MOTION_PRIMITIVE_INPUT_DIVISIONS: int = 10
STEERING_MAX_DIST: float = 10
