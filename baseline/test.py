import sys

if "../" not in sys.path:
  sys.path.append("../")

from epsilon.environment import Environment
from epsilon.building import Building

from LQF import *
from rand import *
from zoning import *

import numpy as np


zoning_main()
lqf_main()
random_main()
