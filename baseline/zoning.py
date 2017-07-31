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
from building import *
from passenger import *
from globals import *
from environment import *
import numpy as np

NUM_EPOCHS = 100


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

    def unload_target_floors(self):
        # find the list of floors the elevator has to go
        return self.dict_passengers.keys()

    def unload_closest_floor(self):
        # find the closest floor to the current location
        dist = [np.abs(self.curr_floor - x) for x in self.unload_target_floors()]
        if dist == []:
            return self.valid_floors[len(self.valid_floors)/2]
        return self.unload_target_floors()[dist.index(min(dist))]

def game_over(env):
    return (env.total_pop >= MAX_POPULATION)

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

def load_closest_floor(call_dict, current_floor, valid_floors, movement_direction):
    # find the closest floor to load passengers
    floors = []
    for f in valid_floors:
        [up, down] = call_dict[f]
        if movement_direction == 1 and up:
            floors.append(f)
        elif movement_direction == -1 and down:
            floors.append(f)
        else:
            pass
    dist = [np.abs(current_floor - x) for x in floors]
    if dist == []:
        return valid_floors[len(valid_floors)/2]
    return floors[dist.index(min(dist))]

def zoning_controller(env):
    action = [0] * NUM_ELEVATORS
    # construct a call info for each floor
    call_dict = dict()
    for floor in env.building.floors:
        call_dict[floor.value] = floor.call
    up_floors = [x for x, v in call_dict.iteritems() if v[0] == 1]
    down_floors = [x for x, v in call_dict.iteritems() if v[1] == 1]
    for i, elevator in enumerate(env.building.elevators):
        if elevator.curr_capacity == 0:
            action[i] = elevator.reset_loc()

        if elevator.move_direction == 0:
            target = elevator.unload_closest_floor()
            action[i] = np.sign(target - elevator.curr_floor)
        else: # elevator is moving
            # decide whether to move to load or move to unload
            f_load = load_closest_floor(call_dict, elevator.curr_floor, elevator.valid_floors, elevator.move_direction)
            f_unload = elevator.unload_closest_floor()
            if np.abs(elevator.curr_floor - f_load) > np.abs(elevator.curr_floor - f_unload):
                # move to unload
                if elevator.curr_floor in elevator.valid_floors:
                    is_upcall = env.building.floors[elevator.curr_floor].call[0]
                    is_downcall = env.building.floors[elevator.curr_floor].call[1]
                    if (is_upcall == elevator.move_direction) or (is_downcall == -elevator.move_direction):
                        action[i] = 0
                    else:
                        action[i] = elevator.move_direction
                else:
                    action[i] = elevator.move_direction
            else:
                # move to load
                action[i] = np.sign(f_load - elevator.curr_floor)
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

        while not game_over(zoning_env):
            zoning_env.populate()
            zoning_env.tic()
            action = zoning_controller(zoning_env)
            s, r = zoning_env.step(action)
        zoning_env.update_global_time_list()
        avg_time = sum(zoning_env.global_time_list) /float(zoning_env.total_pop)
        result += avg_time
    return result / float(epoch)

def zoning_main():
    average_score = zoning_run(epoch=NUM_EPOCHS)
    print("Average Score: {} over {} Epochs".format(average_score, NUM_EPOCHS))

if __name__ == '__main__':
    zoning_main()
