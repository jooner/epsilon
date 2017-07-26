from random import randint
from gloabals import *

class Passenger(object):
    def __init__(self):
        self.destination = None
        self.time = 0
        self.start_floor = randint(0, NUM_FLOORS)
