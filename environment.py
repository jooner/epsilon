import numpy as np

from random import randint, choice
from globals import *
from passenger import *
from building import *

class Environment(object):
    def __init__(self, building):
        self.time = 0
        self.total_pop = 0
        self.curr_pop = 0
        self.building = building
        self.global_time_list = []
        # 0.2 arrivals per sec over 7200 secs (2 hrs)
        self.population_plan = np.random.poisson(0.2, TOTAL_SEC)

    def tic(self): # long live ke$ha
        # TODO: Make this faster without nested for loops
        self.time += 1
        for floor in self.building.floors:
            for passenger in floor.passenger_list:
                passenger.time += 1
        for elevator in self.building.elevators:
            for _, v in elevator.dict_passengers.iteritems():
                for passenger in v:
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
                self.global_time_list.extend(self.building.elevators[i].unload(self.building.floors[flr]))
                self.building.floors[flr] = self.building.elevators[i].load(self.building.floors[flr])

            #print elevator_action
            self.building.elevators[i].move(elevator_action)
            self.building.elevators[i].update()
        self.tic() # progress global time by t += 1
        done = False
        if self.curr_pop >= MAX_POPULATION:
            done = True
        return (self.get_state(), self.get_reward(), done)

    def get_reward(self):
        reward = -sum([e.cumulative_cost for e in self.building.elevators])
        reward -= sum([f.get_cost() for f in self.building.floors])
        return reward

    def get_state(self):
        state = np.zeros(NUM_FLOORS * 2 + NUM_ELEVATORS * 3)
        idx = 0
        for floor in self.building.floors:
            state[idx:idx+2] = floor.call
            idx += 2
        for elevator in self.building.elevators:
            state[idx:idx+3] = [elevator.curr_floor, elevator.move_direction, elevator.curr_capacity]
            idx += 3
        return state

    def populate(self):
        """Populate passenger objects"""
        if self.time <  TOTAL_SEC:
            new_pop = self.population_plan[self.time]
            self.curr_pop += new_pop
            self.total_pop += new_pop
            for _ in xrange(new_pop):
                passenger = Passenger()
                passenger.destination = choice(range(0, passenger.start_floor) +
                                               range(passenger.start_floor + 1, NUM_FLOORS))
                self.building.floors[passenger.start_floor].passenger_list.append(passenger)
                self.building.floors[passenger.start_floor].update_call()

    def update_global_time_list(self):
        for floor in self.building.floors:
            for p in floor.passenger_list:
                self.global_time_list.append(p.time)
        for elev in self.building.elevators:
            for k,v in elev.dict_passengers.iteritems():
                for p in v:
                    self.global_time_list.append(p.time)

    def reset(self):
        building = Building()
        self.__init__(building)
        return self.get_state()
