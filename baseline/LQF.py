"""
Implementation of Baseline Model LQF:

In Longest Queue First, idle elevator cars always prioritize the floor with
the passenger that has waited the longest. When an elevator car is moving,
it will stop at a floor if it has a hall call in the same direction as the car
is traveling. One disadvantage with this is that a phenomenon known as bunching,
where multiple elevator cars arrive at the same floor simultaneously, often
occurs in practice.

"""

from environment import Environment
from building import Building
from globals import *

import numpy as np

NUM_EPOCHS = 100


def lqf_controller(env):
    action = [None] * NUM_ELEVATORS
    priority_q = {} # important => lower q key
    for floor in env.building.floors:
        priority_q[-floor.get_cost()] = floor.value
    for i, elevator in enumerate(env.building.elevators):
        if elevator.move_direction == 0:
            priority_flr = priority_q[min(priority_q.keys())]
            action[i] = np.sign(priority_flr - elevator.curr_floor)
        else: # elevator is moving
            ## NOTE: Doesn't consider when to stop for loading -- only considers when to
            ## stop for unloading.
            is_upcall = env.building.floors[elevator.curr_floor].call[0]
            is_downcall = env.building.floors[elevator.curr_floor].call[1]
            if (is_upcall == elevator.move_direction) or (is_downcall == -elevator.move_direction):
                action[i] = 0
            else:
                action[i] = elevator.move_direction
    return action


def game_over(env):
    return (env.total_pop >= MAX_POPULATION)

def lqf_run(epoch=1):
    result = 0
    for i in range(epoch):
        lqf_building = Building()
        lqf_env = Environment(lqf_building)
        #print "---------------------%d"%i

        while not game_over(lqf_env):
            #print lqf_env.total_pop
            lqf_env.populate() # populate the building
            lqf_env.tic() # t += 1
            action = lqf_controller(lqf_env)
            s, r, _ = lqf_env.step(action)
            #print "action = %s \t reward = %f"%(action, r)
        #scores.append(lqf_env.get_reward())
        lqf_env.update_global_time_list()
        avg_time = sum(lqf_env.global_time_list) / float(lqf_env.total_pop)
        result += avg_time
    return result / float(epoch)

def lqf_main():
    average_score = lqf_run(epoch=NUM_EPOCHS)
    print("[LQF Baseline]\tAverage Score: {} over {} Epochs".format(average_score, NUM_EPOCHS))

if __name__ == "__main__":
    lqf_main()
