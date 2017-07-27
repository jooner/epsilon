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
            diff = passenger.start_floor - passenger.destination
            if diff > 0:
                self.call[1] = 1
            elif diff < 0:
                self.call[0] = 1
            else:
                pass

    def get_cost(self):
        return sum([p.time for p in self.passenger_list])

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
        for i, p in enumerate(floor.passenger_list):
            if self.curr_capacity >= MAX_CAP_ELEVATOR:
                break
            if p.destination not in self.dict_passengers.keys():
                self.dict_passengers[p.destination] = [p]
            else:
                self.dict_passengers[p.destination].append(p)
            del_list.append(i)
            self.curr_capacity += 1
            self.cumulative_cost += p.time
        for idx in del_list:
            floor.passenger_list.pop(idx)
        # reinitialize floor calls
        floor.call = [0,0]
        floor.update_call()

        return floor

    def unload(self, floor):
        """
        Given Floor object, unload the passengers
        """
        if floor.value in self.dict_passengers.keys():
            for x in self.dict_passengers[floor.value]:
                self.cumulative_cost -= x.time
                self.curr_capacity -= 1
            self.dict_passengers.pop(floor.value, None)



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
        self.cumulative_cost += self.curr_capacity
