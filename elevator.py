from globals import *

"""
Elevator Module
"""
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
        self.cummulative_cost = 0

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
        Load the passengers
        """
        for p in floor.passenger_list:
            if self.curr_capacity >= MAX_CAP_ELEVATOR:
                pass
            if p.destination not in self.dict_passengers.keys():
                self.dict_passengers[p.destination] = [p]
            else:
                self.dict_passengers[p.destination].append(p)
            self.curr_capacity += 1
            self.cummulative_cost += p.time

    def unload(self, floor):
        """
        Unload the passengers
        """
        if floor.value in self.dict_passengers.keys():
            cost = sum([x.time for x in self.dict_passengers[floor.value]])
            self.dict_passengers.pop(floor.value, None)
            self.cummulative_cost -= cost

    def update(self):
        """
        Call this function at the end of each iteration
        """
        # Update the current floor
        if self.move_direction == 1:
            if self.curr_floor < NUM_FLOORS-1:
                self.curr_floor += 1
        elif self.move_direction == -1:
            if self.curr_floor > 0:
                self.curr_floor -= 1

        # Update the cost
        self.cummulative_cost += self.curr_capacity



