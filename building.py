from globals import *

class Building(object):
    def __init__(self):
        self.floors = []
        for _ in xrange(NUM_FLOORS):
            self.floors.append(Floor())

class Floor(object):
    def __init__(self):
        self.passenger_list = [] # list of passenger objects
        self.is_elevator = False # boolean
        self.call = [0,0]
