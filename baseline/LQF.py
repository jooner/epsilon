"""
Implementation of Baseline Model LQF:

In Longest Queue First, idle elevator cars always prioritize the floor with
the passenger that has waited the longest. When an elevator car is moving,
it will stop at a floor if it has a hall call in the same direction as the car
is traveling. One disadvantage with this is that a phenomenon known as bunching,
where multiple elevator cars arrive at the same floor simultaneously, often
occurs in practice.

"""
import sys

if "../" not in sys.path:
  sys.path.append("../")

from epsilon.altenv import Environment
from epsilon.building import Building
from epsilon.globals import *

import numpy as np


def lqf_controller(env):

    def closest_elevator(elevators, target_floor, idx):
        # find the closest elevator to the target floor
        dist = []
        for i, elevator in enumerate(elevators):
            if i not in idx:
                dist.append(np.inf)
            else:
                dist.append(np.abs(elevator.curr_floor - target_floor))
        return dist.index(min(dist))

    action = [None] * NUM_ELEVATORS
    stoplist = env.elevators_to_stop() # indices of elevators to stop
    # find available elevators and stop elevators if they are on target floor
    for i, elevator in enumerate(env.building.elevators):
        if i in stoplist: # for unloading
            action[i] = 0
            continue
        is_upcall = env.building.floors[elevator.curr_floor].call[0]
        is_downcall = env.building.floors[elevator.curr_floor].call[1]
        if (is_upcall == 1 and elevator.move_direction==1) or (is_downcall == 1 and elevator.move_direction == -1):
            action[i] = 0
        elif elevator.move_direction == 0 and elevator.curr_capacity == 0:
            pass
        elif elevator.move_direction == 0:
            action[i] = np.sign(elevator.dict_passengers.keys()[0] - elevator.curr_floor)
        else: # if elevator is moving it let the elevator maintain status quo
            action[i] = elevator.move_direction
    # if elevator is NOT moving, send the closest one to high priority floor

    if None in action:
        priority_q = {} # important => lower q key
        idx = [i for i in xrange(len(action)) if action[i] == None]
        for floor in env.building.floors:
            priority_q[-floor.get_cost()] = floor.value
        pq = sorted(priority_q)[:sum([1 for i in action if i == None])]
        for p in pq:
            target_floor = priority_q[p]
            e_i = closest_elevator(env.building.elevators, target_floor, idx)
            action[e_i] = np.sign(target_floor - env.building.elevators[e_i].curr_floor)

    if None in action:
        for i, a in enumerate(action):
            if a == None:
                action[i] = 0
    return action

def lqf_run(epoch=1):
    result = 0
    rewards = []
    for i in range(epoch):
        lqf_building = Building()
        lqf_env = Environment(lqf_building)
        #print "---------------------%d"%i
        lqf_env.get_state()
        episode_reward = 0
        while not lqf_env.is_done():
            #print lqf_env.total_pop
            action = lqf_controller(lqf_env)
            s, r, _ = lqf_env.step(action)
            episode_reward += r
            #print "action = %s \t reward = %f"%(action, r)
        #scores.append(lqf_env.get_reward())
        lqf_env.update_global_time_list()
        rewards.append(episode_reward)
        avg_time = sum(lqf_env.global_time_list) / float(lqf_env.total_pop)
        result += avg_time
    return result / float(epoch), float(sum(rewards)) / len(rewards)

def lqf_main():
    average_score, avg_reward = lqf_run(epoch=NUM_EPOCHS)
    print("[LQF Baseline]\t\tAverage Score: {} over {} Epochs".format(average_score, NUM_EPOCHS))
    print("[LQF Baseline]\t\tAverage Reward: {} over {} Epochs".format(avg_reward, NUM_EPOCHS))

if __name__ == "__main__":
    lqf_main()
