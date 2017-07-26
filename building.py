from globals import *
from elevator import Elevator

class Building(object):
    def __init__(self):
        self.floors = []
        for i in xrange(NUM_FLOORS):
            floor = Floor(i)
            self.floors.append(floor)
        self.elevators = []
        for _ in xrange(NUM_ELEVATORS):
            self.elevators.append(Elevator())

class Floor(object):
    def __init__(self, floor):
        self.value = floor
        self.passenger_list = [] # list of passenger objects
        self.is_elevator = False # boolean
        self.call = [0,0]
