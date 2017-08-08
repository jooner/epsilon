import numpy as np

from globals import *

class Building(object):
    def __init__(self):
        self.floors = []
        self.elevators = []

        for i in xrange(NUM_FLOORS):
            floor = Floor(i)
            self.floors.append(floor)

        for _ in xrange(NUM_ELEVATORS):
            self.elevators.append(Elevator())

class Floor(object):
    def __init__(self, floor):
        self.value = floor
        self.passenger_list = [] # list of passenger objects
        self.is_elevator = False # boolean
        self.call = [0,0] # [is_up, is_down]

    def update_call(self):
        for passenger in self.passenger_list:
            diff = passenger.destination - passenger.start_floor
            if diff > 0:
                self.call[1] = 1
            elif diff < 0:
                self.call[0] = 1
            else:
                pass

    def get_cost(self):
        return sum([(p.time**2) for p in self.passenger_list])

class Elevator(object):
    def __init__(self):
        # dictionary of passengers (key: destination floor, value: passenger object)
        self.dict_passengers = dict()
        # current floor the elevator is at
        self.curr_floor = 0
        # movement direction (0: stop, 1: up, -1: down)
        self.move_direction = 0
        # current number of passengers on the elevator
        self.curr_capacity = 0
        # total cost incurred so far
        self.cumulative_cost = 0

    def move(self, action):
        if action == 0:
            self.move_direction = action
        elif action == 1:
            if self.move_direction != -1:
                self.move_direction = action
        elif action == -1:
            if self.move_direction != 1:
                self.move_direction = action
        else:
            raise ValueError("action value should be one of 0, 1, -1")

    def load(self, floor):
        """
        Given Floor object, load the passengers
        """
        del_list = [] # idices of passenger objects to remove
        if self.curr_capacity < MAX_CAP_ELEVATOR and floor.passenger_list != []:
            if self.curr_capacity == 0:
                closest_passenger = np.argmin([abs(p.destination - floor.value) for p in floor.passenger_list])
                mov_dir = floor.passenger_list[closest_passenger].get_direction()
            else:
                mov_dir = np.sign(self.dict_passengers.keys()[0] - floor.value)
            for i, p in enumerate(floor.passenger_list):
                if self.curr_capacity != MAX_CAP_ELEVATOR and p.get_direction() == mov_dir:
                    if p.destination not in self.dict_passengers.keys():
                        self.dict_passengers[p.destination] = [p]
                    else:
                        self.dict_passengers[p.destination].append(p)
                    print "loaded!"
                    del_list.append(i)
                    self.curr_capacity += 1
                    self.cumulative_cost += p.time ** 2 - p.time
            temp = []
            for i in xrange(len(floor.passenger_list)):
                if i not in del_list:
                    temp.append(floor.passenger_list[i])
            floor.passenger_list = temp
            # reinitialize floor calls
            floor.call = [0,0]
            floor.update_call()
        return floor

    def unload(self, floor):
        """
        Given Floor object, unload the passengers
        """
        total_time = []
        if floor.value in self.dict_passengers.keys():
            for p in self.dict_passengers[floor.value]:
                # reward when unloading person
                self.cumulative_cost -= p.time ** 2 + p.time
                self.curr_capacity -= 1
                total_time.append(p.time)
            self.dict_passengers.pop(floor.value, None)
            print "unloaded!"
        return total_time

    def update(self):
        """
        Call this function at the end of each iteration
        """
        # Update the current floor
        if self.move_direction == 1:
            if self.curr_floor == NUM_FLOORS - 1:
                self.move_direction = 0
            if self.curr_floor < NUM_FLOORS - 1:
                self.curr_floor += 1
        elif self.move_direction == -1:
            if self.curr_floor == 0:
                self.move_direction = 0
            if self.curr_floor > 0:
                self.curr_floor -= 1

        # Update the cost
        for passengers in self.dict_passengers.values():
            self.cumulative_cost += sum([(p.time ** 2) for p in passengers])
