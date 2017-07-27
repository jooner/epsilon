import numpy as np

from random import randint, choice
from globals import *


class Environment(object):
    def __init__(self, building):
        self.time = 0
        self.total_pop = 0
        self.curr_pop = 0
        self.building = building

    def tic(self): # long live ke$ha
        for floor in self.building.floors:
            for passenger in floor.passenger_list:
                passenger.time += 1
        for elevator inself.building.elevators:
            for passenger in elevator.values():
                passenger.time += 1

    def step(self, a):
        """
        Input: action [E1, E2, E3] where En is one of 1,0,-1
        Returns: next state and reward
        """
        assert len(a) == len(self.building.elevators)
        for i, elevator_action in enumerate(a):
            if elevator_action == 0:
                flr = self.building.elevators[i].curr_floor
                self.building.elevators[i].unload(self.building.floors[flr])
                self.building.floors[flr] = self.building.elevators[i].load(self.building.floors[flr])
            self.building.elevators[i].move(elevator_action)
            self.building.elevators[i].update()
        reward = -sum([x.cumulative_cost for x in self.building.elevators])
        self.tic() # progress global time by t += 1
        return (self.get_state, reward)

    def get_state(self):
        state = np.zeros(NUM_FLOORS * 2 + NUM_ELEVATORS * 3)
        idx = 0
        for floor in self.buildings.floors:
            state[idx:idx+1] = floor.call
            idx += 2
        for elevator in self.buildings.elevators:
            state[idx:idx+2] = [elevator.curr_floor, elevator.move_direction, elevator.curr_capacity]
            idx += 3
        return state

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
