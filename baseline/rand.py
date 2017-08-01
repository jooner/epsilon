"""
Implementation of Baseline Model Random:

Randomly assign the elevators to random floors.

"""
import sys

if "../" not in sys.path:
  sys.path.append("../")

from epsilon.environment import *
from epsilon.building import *
from epsilon.globals import *

import numpy as np

NUM_EPOCHS = 100


def random_controller(env):
    action = [np.random.randint(3) - 1 for i in range(len(env.building.elevators))]
    return action


def random_run(epoch=1):
    result = 0
    for i in range(epoch):
        random_building = Building()
        random_env = Environment(random_building)
        #print "---------------------%d"%i

        while not random_env.is_done():
            random_env.populate() # populate the building
            random_env.tic() # t += 1
            action = random_controller(random_env)
            s, r, _ = random_env.step(action)
        random_env.update_global_time_list()
        avg_time = sum(random_env.global_time_list) / float(random_env.total_pop)
        result += avg_time
    return result / float(epoch)

def random_main():
    average_score = random_run(epoch=NUM_EPOCHS)
    print("[Random Baseline]\tAverage Score: {} over {} Epochs".format(average_score, NUM_EPOCHS))

if __name__ == "__main__":
    random_main()
