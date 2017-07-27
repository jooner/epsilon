"""
Implementation of Baseline Model LQF:

In Longest Queue First, idle elevator cars always prioritize the floor with
the passenger that has waited the longest. When an elevator car is moving,
it will stop at a floor if it has a hall call in the same direction as the car
is traveling. One disadvantage with this is that a phenomenon known as bunching,
where multiple elevator cars arrive at the same floor simultaneously, often
occurs in practice.

"""
from __future__ import absolute_import
from __future__ import print_function

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
            is_upcall = env.building.floors[elevator.curr_floor].call[0]
            is_downcall = env.building.floors[elevator.curr_floor].call[1]
            if (is_upcall == elevator.move_direction) or (is_downcall == -elevator.move_direction):
                action[i] = 0
            else:
                action[i] = elevator.move_direction
    return action


def game_over(env):
    return (env.total_pop > MAX_POPULATION)

def run(lqf_env, epoch=1):
    scores = []
    for _ in range(epoch):
        while not game_over(lqf_env):
            lqf_env.populate() # populate the building
            lqf_env.tic() # t += 1
            action = lqf_controller(lqf_env)
            lqf_env.step(action)
        scores.append(lqf_env.get_reward())
    # average performance over epochs
    return sum(scores) / len(scores)

def main():
    lqf_building = Building()
    lqf_env = Environment(lqf_building)
    average_score = run(lqf_env, epoch=NUM_EPOCHS)
    print("Average Score: {} over {} Epochs".format(average_score, NUM_EPOCHS))

if __name__ == "__main__":
    main()
