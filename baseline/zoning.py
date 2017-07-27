from building import *
from passenger import *
from globals import *
from environment import *

"""
Zoning Baseline
"""

class Elevator_Z(Elevator):
    def __init__(self, no, zones):
        self.valid_floors = zones[no]
        self.curr_floor = self.valid_floors[len(self.valid_floors)/2]

    def move(self):
        pass

def _partition(lst, n):
    # Helper function: partition a lst into n approximately-equally-sized lists
    division = len(lst) / float(n)
    return np.asarray([ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n) ])

def divide_simple_zones():
    # simply ordered zone
    return _partition(range(NUM_FLOORS), NUM_ELEVATORS)

def divide_random_zones():
    # randomly shuffled zone
    floors = list(np.random.permutation(NUM_FLOORS))
    return _partition(floors, NUM_ELEVATORS)

def main():
    zones = divide_random_zones()
    for z in zones:
        z.sort()
    building = Building()
    for i, e in enumerate(building.elevators):
        e = Elevator_Z(i, zones)
        e = zones[i][len(zones)/2]

if __name__ == '__main__':
    main()
