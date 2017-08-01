from environment import Environment
from building import Building
from globals import *

from LQF import *
from rand import *
from zoning import *

import numpy as np

NUM_EPOCHS = 100

# Override the global variables from testing purposes
NUM_ELEVATORS = 2
NUM_FLOORS = 4
MAX_CAP_ELEVATOR = 1
MAX_POPULATION = 3

"""
zoning_main()
lqf_main()
random_main()
"""

def game_over(env):
    return (env.total_pop >= MAX_POPULATION)

def run(epoch=1, controller):
    result = 0
    for i in range(epoch):
        building = Building()
        env = Environment(building)

        while not game_over(env):
            random_env.populate() # populate the building
            random_env.tic() # t += 1
            action = controller(env)
            s, r, _ = env.step(action)
        env.update_global_time_list()
        avg_time = sum(env.global_time_list) / float(env.total_pop)
        result += avg_time
    return result / float(epoch)
