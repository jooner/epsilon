import numpy as np

from random import randint, choice
from globals import *


class Environment(object):
    def __init__(self, building):
        self.time = 0
        self.total_pop = 0
        self.curr_pop = 0
        self.building = building

    def step(self, a):
        """
        Input: action
        Returns: next state and reward
        """
        pass

    def populate(self):
        """Populate passenger objects"""
        if self.total_pop < MAX_POPULATION:
            new_pop = randint(0, 5)
            self.curr_pop += new_pop
            self.total_pop += new_pop
            for _ in xrange(new_pop):
                passenger = Passenger()
                passenger.destination = choice(range(0, self.start_floor) +
                                               range(self.start_floor + 1, NUM_FLOORS))
                self.building.floors[passenger.start_floor].passenger_list.append(passenger)
                self.building.floors[passenger.start_floor].update_call()
