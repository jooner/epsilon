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
        self.old_state = np.zeros((NUM_ELEVATORS * 3 + 2, NUM_FLOORS, NUM_FLOORS))
        self.old_old_state = np.zeros((NUM_ELEVATORS * 3 + 2, NUM_FLOORS, NUM_FLOORS))
        self.populate()
        # 0.2 arrivals per sec over 7200 secs (2 hrs)
        #self.population_plan = np.random.poisson(0.2, TOTAL_SEC)

    def tic(self):
        # TODO: Make this faster without nested for loops
        #self.populate()
        for floor in self.building.floors:
            for passenger in floor.passenger_list:
                passenger.time += 1
        for elevator in self.building.elevators:
            for _, v in elevator.dict_passengers.iteritems():
                for passenger in v:
                    passenger.time += 1
        self.time += 1

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
                k = len(self.global_time_list)
                self.global_time_list.extend(self.building.elevators[i].unload(self.building.floors[flr]))
                #if k != len(self.global_time_list):
                #    print self.global_time_list
                self.building.floors[flr] = self.building.elevators[i].load(self.building.floors[flr])

            #print elevator_action
            self.building.elevators[i].move(elevator_action)
            self.building.elevators[i].update()
        self.tic() # progress global time by t += 1

        return (self.get_state(), self.get_reward(), self.is_done())

    def is_done(self):
        return (self.time > TOTAL_SEC)

    def get_reward(self):
        # passenger wait time cost
        p_cost1 = sum([e.cumulative_cost for e in self.building.elevators])
        p_cost2 = sum([f.get_cost() for f in self.building.floors])
        # elevator movement amount cost
        e_cost = sum([e.movement for e in self.building.elevators])
        reward = -(p_cost1 + p_cost2 + e_cost)
        #print(p_cost1, p_cost2, e_cost, reward)
        return reward

    def get_state(self):
        state = np.zeros((NUM_ELEVATORS * 3 + 2, NUM_FLOORS, NUM_FLOORS))
        for floor in self.building.floors:
            for passenger in floor.passenger_list:
                state[0, floor.value, passenger.destination] += 1
            for passenger in floor.passenger_list:
                state[1, floor.value, passenger.destination] += passenger.time
        for j, elevator in enumerate(self.building.elevators):
            state[3*j+2, elevator.curr_floor, :] = [1] * NUM_FLOORS # note where the elevator is
            for destination, passenger_list in elevator.dict_passengers.iteritems():
                state[3*j+3, elevator.curr_floor, destination] += len(passenger_list)
                state[3*j+4, elevator.curr_floor, destination] += sum([p.time for p in passenger_list])
        concat_state = np.concatenate((self.old_old_state[...,np.newaxis], self.old_state[...,np.newaxis]), axis=3)
        self.old_old_state = self.old_state
        concat_state = np.concatenate((concat_state, state[...,np.newaxis]), axis=3)
        self.old_state = state
        return concat_state

    def populate(self):
        """Populate passenger objects. Hard Code the numbers. yay.
        Experiments should be run with num_floor = 3, num_elev = 2, cap = 2."""
#        self.state = np.zeros((NUM_FLOORS, NUM_FLOORS, 2))
#        self.state[0, 2, 0] += 1
#        self.state[0, 1, 0] += 1
#        self.state[2, 3, 0] += 1
#        self.state[2, 0, 0] += 1
#        self.state[3, 1, 0] += 1

        #start_dest_pairs = [(0,2), (0,1), (2,0), (3,1), (2,3)]
        start_dest_pairs = []
        for i in xrange(5):
            s = np.random.randint(4)
            d = np.random.randint(4)
            if s == d:
                d = (d + 1) % 4
            start_dest_pairs.append((s, d))
        self.curr_pop += 5
        self.total_pop += 5

        for pair in start_dest_pairs:
            passenger = Passenger()
            passenger.start_floor = pair[0]
            passenger.destination = pair[1]
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
