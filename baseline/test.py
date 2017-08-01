import sys

if "../" not in sys.path:
  sys.path.append("../")

from epsilon.environment import Environment
from epsilon.building import Building
from epsilon.globals import *

from LQF import *
from rand import *
from zoning import *

import numpy as np

# Override the global variables from testing purposes
NUM_ELEVATORS = 3
NUM_FLOORS = 30
MAX_CAP_ELEVATOR = 20
TOTAL_SEC = 1800
NUM_EPOCHS = 10

zoning_main()
lqf_main()
random_main()
