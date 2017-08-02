"""
Implementation of Baseline Model Zoning.

A building with N floors is divided into m zones, where m is the number of elevator cars.
Each zone get at least int(N/m) floors and one elevator car. If the number of zones is not
divisible by the number of floors, N mod m zones will get one additional floor.
The boundaries of the zones are static, which means that the floors assigned to a
specific zone are not changed during the course of the simulation. The zones do only
consider the arrival floor of a passenger and not the destination. This makes the zoning
approach only viable in down-peak scenarios. When an elevator car is idle, it is sent to
the floor in the middle of its zone. When a car is moving, it can pick up passengers in
the same movement direction inside its zone.

"""
import sys

if "../" not in sys.path:
  sys.path.append("../")

from epsilon.building import *
from epsilon.passenger import *
from epsilon.globals import *
from epsilon.environment import *
import numpy as np


class Building_z():
    def __init__(self, zones):
        self.floors = []
        self.elevators = []

        for i in xrange(NUM_FLOORS):
            floor = Floor(i)
            self.floors.append(floor)

        for i in xrange(NUM_ELEVATORS):
            self.elevators.append(Elevator_z(i, zones))

class Elevator_z(Elevator):
    def __init__(self, no, zones):
        super(Elevator_z, self).__init__()
        self.valid_floors = zones[no]
        self.move_direction = self.reset_loc()

    # Call when there are no one on the elevator -- to move the elevator to the middle
    # of the zone
    def reset_loc(self):
        target_floor = self.valid_floors[len(self.valid_floors)/2]
        return np.sign(target_floor - self.curr_floor)

    def load(self, floor):
        del_list = [] # idices of passenger objects to remove
        # load the passengers only on the valid floors
        if floor.value in self.valid_floors:
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
            temp = []
            for i in xrange(len(floor.passenger_list)):
                if i not in del_list:
                    temp.append(floor.passenger_list[i])
            floor.passenger_list = temp
            # reinitialize floor calls
            floor.call = [0,0]
            floor.update_call()

        return floor

def _partition(lst, n):
    # Helper function: partition a lst into n approximately-equally-sized lists
    division = len(lst) / float(n)
    return np.asarray([lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n)])

def divide_simple_zones():
    # simply ordered zone
    return _partition(range(NUM_FLOORS), NUM_ELEVATORS)

def divide_random_zones():
    # randomly shuffled zone
    floors = list(np.random.permutation(NUM_FLOORS))
    return _partition(floors, NUM_ELEVATORS)

def zoning_controller(env):
    action = [None] * NUM_ELEVATORS
    stoplist = env.elevators_to_stop()
    for i, elevator in enumerate(env.building.elevators):
        if i in stoplist:
            action[i] = 0
            continue

        is_upcall = env.building.floors[elevator.curr_floor].call[0]
        is_downcall = env.building.floors[elevator.curr_floor].call[1]
        if (is_upcall == 1 and elevator.move_direction==1 and elevator.curr_floor in elevator.valid_floors) or \
        (is_downcall == 1 and elevator.move_direction == -1 and elevator.curr_floor in elevator.valid_floors):
            action[i] = 0 # stopping elevator for loading

        elif elevator.move_direction == 0 and elevator.curr_capacity == 0:
            midzone = elevator.valid_floors[len(elevator.valid_floors)/2]
            action[i] = np.sign(midzone - elevator.curr_floor)

        elif elevator.move_direction == 0:
            action[i] = np.sign(elevator.dict_passengers.keys()[0] - elevator.curr_floor)
        else: # if elevator is moving it let the elevator maintain status quo
            action[i] = elevator.move_direction

    return action

def zoning_run(epoch=1, zone_type="random"):
    result = 0
    for i in range(epoch):
        if zone_type == "random":
            zones = divide_random_zones()
        elif zone_type == "simple":
            zones = divide_simple_zones()
        else:
            raise ValueError("Input zone_type [%s] is not defined."%zone_type)

        # Sort the floors in each zone
        _ = [x.sort() for x in zones]
        building = Building_z(zones)
        zoning_env = Environment(building)

        while not zoning_env.is_done():
            zoning_env.populate()
            zoning_env.tic()
            action = zoning_controller(zoning_env)
            s, r, _ = zoning_env.step(action)
        zoning_env.update_global_time_list()
        avg_time = sum(zoning_env.global_time_list) /float(zoning_env.total_pop)
        result += avg_time
    return result / float(epoch)

def zoning_main():
    average_score = zoning_run(epoch=NUM_EPOCHS)
    print("[Zoning Baseline]\tAverage Score: {} over {} Epochs".format(average_score, NUM_EPOCHS))

if __name__ == '__main__':
    zoning_main()
