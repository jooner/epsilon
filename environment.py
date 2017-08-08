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
        self.old_state = np.zeros([NUM_FLOORS, (MAX_CAP_ELEVATOR) + NUM_VALID_ACTIONS + 4 + 1])
        # 0.2 arrivals per sec over 7200 secs (2 hrs)
        self.population_plan = np.random.poisson(0.3, TOTAL_SEC)

    def tic(self): # long live ke$ha
        # TODO: Make this faster without nested for loops
        self.time += 1
        self.populate()
        for floor in self.building.floors:
            for passenger in floor.passenger_list:
                passenger.time += 1
        for elevator in self.building.elevators:
            for _, v in elevator.dict_passengers.iteritems():
                for passenger in v:
                    passenger.time += 1

    def elevators_to_stop(self):
        # returns a list of elevator indices that need to stop
        stoplist = []
        for i, e in enumerate(self.building.elevators):
            if e.curr_capacity != 0:
                for dest, _ in e.dict_passengers.iteritems():
                    if e.curr_floor ==  dest:
                        stoplist.append(i)
        return stoplist

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
        print self.get_state(), self.get_reward()
        return (self.get_state(), self.get_reward(), self.is_done())

    def is_done(self):
        return (self.time > TOTAL_SEC)

    def get_reward(self):
        reward = -sum([e.cumulative_cost for e in self.building.elevators])
        reward -= sum([f.get_cost() for f in self.building.floors])
        return reward / float(1e8)

        """
        try:
            return (-sum(self.global_time_list) / float(len(self.global_time_list))) /float(1e3)
        except:
            return 0
        """

    def get_state(self):
        state = np.zeros([NUM_FLOORS, 4 + MAX_CAP_ELEVATOR + NUM_VALID_ACTIONS + 1])
        for i, floor in enumerate(self.building.floors):
            nonzero_idx = floor.call[0] + floor.call[1] * 2
            state[i][nonzero_idx] = 1

        for e_i, elevator in enumerate(self.building.elevators):
            part_state = np.zeros((MAX_CAP_ELEVATOR) + NUM_VALID_ACTIONS + 1)
            part_state[:elevator.curr_capacity] = [1] * elevator.curr_capacity
            part_state[MAX_CAP_ELEVATOR:][elevator.move_direction+1] = 1
            min_dist = NUM_FLOORS
            for dest_flr in elevator.dict_passengers.keys():
                min_dist = min(min_dist, abs(elevator.curr_floor - dest_flr))
            part_state[-1] = min_dist
            state[elevator.curr_floor][4:] = part_state
        channeled_state = np.zeros((2, state.shape[0], state.shape[1]))
        channeled_state[0, :, :] = self.old_state
        channeled_state[1, :, :] = state
        self.old_state = state
        return state

    def populate(self):
        """Populate passenger objects"""
        if self.time < TOTAL_SEC:
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
        # TODO: make this faster by removing nested for loop
        for floor in self.building.floors:
            for p in floor.passenger_list:
                self.global_time_list.append(p.time)
        for elev in self.building.elevators:
            for _,v in elev.dict_passengers.iteritems():
                for p in v:
                    self.global_time_list.append(p.time)

    def reset(self):
        building = Building()
        self.__init__(building)
        return self.get_state()
