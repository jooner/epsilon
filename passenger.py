from random import randint
from globals import *

import numpy as np

class Passenger(object):
    def __init__(self):
        self.time = 0
        self.start_floor = 0
        #self.start_floor = randint(0, NUM_FLOORS-1)
        self.destination = None

    def get_direction(self):
        if self.destination != None:
            return np.sign(self.destination - self.start_floor)
