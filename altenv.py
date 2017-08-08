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
        self.populate()
        # 0.2 arrivals per sec over 7200 secs (2 hrs)
        #self.population_plan = np.random.poisson(0.2, TOTAL_SEC)

    def tic(self): # long live ke$ha
        # TODO: Make this faster without nested for loops
        self.time += 1
        #self.populate()
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
        return (self.get_state(), self.get_reward(), self.is_done())

    def is_done(self): # TODO
        return (self.time > TOTAL_SEC)

    def get_reward(self):
        reward = -sum([e.cumulative_cost for e in self.building.elevators])
        reward -= sum([f.get_cost() for f in self.building.floors])
        return reward / float(1e8)

    def get_state(self):  # TODO
        state = np.zeros((NUM_ELEVATORS * 2 + 1, NUM_FLOORS, NUM_FLOORS))
        for i, floor in enumerate(self.building.floors):
            for passenger in floor.passenger_list:
                state[0, floor.value, passenger.destination] += 1
        for j, elevator in enumerate(self.building.elevators):
            for destination, passenger_list in elevator.dict_passengers.iteritems():
                state[2*j+1, elevator.curr_floor, destination] += len(passenger_list)
                state[2*j+2, elevator.curr_floor, destination] += sum([p.time for p in passenger_list])
        return state


    def populate(self):  # TODO
        """Populate passenger objects. Hard Code the numbers. yay.
        Experiments should be run with num_floor = 3, num_elev = 2, cap = 2."""
#        self.state = np.zeros((NUM_FLOORS, NUM_FLOORS, 2))
#        self.state[0, 2, 0] += 1
#        self.state[0, 1, 0] += 1
#        self.state[2, 3, 0] += 1
#        self.state[2, 0, 0] += 1
#        self.state[3, 1, 0] += 1

        start_dest_pairs = [(0,2), (0,1), (2,0), (3,1), (2,3)]

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
