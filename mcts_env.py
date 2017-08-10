from mcts import *

UCB_BANDIT = 1

class MCTSWorldTree(object):
    def __init__(self, state, reward):
        self.root = StateNode(None, state, reward)
        self.state = self.root
        self.trajectory = []
        self.mcts = MCTS(tree_policy=UCB1(c=UCB_BANDIT),
                         default_policy=immediate_reward,
                         backup=monte_carlo)

    def perform(self):

        return self.mcts(self)
        #self.state.children[action]

    def real_perform(self, action):
        pass
