from altenv import Environment

import numpy as np
from scipy.stats import rv_discrete, entropy
from copy import deepcopy

class ElevatorWorldAction(object):
    def __init__(self, actions):
        self.actions = actions

class ElevatorWorld(object):
    def __init__(self, world_state, belief=None):
        self.world = world_state # get get_state from altenv
        # hard-code actions for now
        self.actions = [np.asarray([-1,-1]), np.asarray([-1,0]), np.asarray([-1,1])
                        np.asarray([0,-1]), np.asarray([0,0]), np.asarray([0,1])
                        np.asarray([1,-1]), np.asarray([1,0]), np.asarray([1,1])]
        if belief:
            self.belief = belief
        else:
            self.belief =  dict((a, np.array([1] * 9)) for a in self.actions)

    def perform(self, action):
        probs = self.belief[action] / np.sum(self.belief[action])
        distrib = rv_discrete(values=(range(len(probs)), probs))


        # draw sample
        sample = distrib.rvs()

        # update belief accordingly
        belief = deepcopy(self.belief)
        belief[action][sample] += 1

        # manual found
        if (self.pos == self.world.manual).all():
            print("m", end="")
            belief = {ToyWorldAction(np.array([0, 1])): [50, 1, 1, 1],
                      ToyWorldAction(np.array([0, -1])): [1, 50, 1, 1],
                      ToyWorldAction(np.array([1, 0])): [1, 1, 50, 1],
                      ToyWorldAction(np.array([-1, 0])): [1, 1, 1, 50]}

        # build next state
        pos = self._correct_position(self.pos + self.actions[sample].action)

        return ToyWorldState(pos, self.world, belief)
