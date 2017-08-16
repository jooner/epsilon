import numpy as np

POSS_ACTIONS = 9 # NUM_VALID_ACTIONS ** NUM_ELEVATORS

# Node Classes
class Node(object):
    def __init__(self, parent, dream=None):
        self.parent = parent
        self.children = {}
        self.q = 0
        self.n = 0
        self.dream = dream

class StateNode(Node):
    """
    A node holding a state in the tree.
    """
    def __init__(self, parent, state, reward):
        super(StateNode, self).__init__(parent)
        self.state = state
        self.reward = reward
        self.untried_actions = range(POSS_ACTIONS)

    def is_terminal():
        return self.children == {}

    def act(action):
        self.dream.step(action)
        return self.dream.get_state()

class ActionNode(Node):
    """
    A node holding an action in the tree.
    """
    def __init__(self, parent, action):
        super(ActionNode, self).__init__(parent)
        self.action = action
        self.n = 0 # num times it been explored

    def sample_state():
        return self.parent.act(self.action)


# Main World
class MCState(object):
    def __init__(self, state, reward, env):
        self.root = StateNode(Node(none), state, reward)
        self.dream = env


# Tree Policy
class UCB1(object):
    """
    The typical bandit upper confidence bounds algorithm.
    """
    def __init__(self, c):
        self.c = c

    def __call__(self, action_node):
        if self.c == 0:  # assert that no nan values are returned
                         # for action_node.n = 0
            return action_node.q

        return (action_node.q +
                self.c * np.sqrt(2 * np.log(action_node.parent.n) /
                                 action_node.n))

# Default policy
def immediate_reward(tree):
    """
    Estimate the reward with the immediate return of that state.
    :param state_node:
    :return:
    """
    return tree.state.reward(tree.state.parent.parent.state,
                             tree.state.parent.action)

# Monte Carlo Backup
def monte_carlo(node):
    """
    A monte carlo update as in classical UCT.
    See feldman amd Domshlak (2014) for reference.
    :param node: The node to start the backup from
    """
    r = node.reward
    while node is not None:
        node.n += 1
        node.q = ((node.n - 1) / node.n) * node.q + 1 / node.n * r
        node = node.parent


# MCTS(tree_policy=UCB1, default_policy=immediate_reward, backup=monte_carlo)
class MCTS(object):
    """
    The central MCTS class, which performs the tree search. It gets a
    tree policy, a default policy, and a backup strategy.
    See e.g. Browne et al. (2012) for a survey on monte carlo tree search
    """
    def __init__(self, tree_policy, default_policy, backup):
        self.tree_policy = tree_policy
        self.default_policy = default_policy
        self.backup = backup

    def __call__(self, tree, n=15):
        """
        Run the monte carlo tree search.
        :param root: The StateNode
        :param n: The number of roll-outs to be performed
        :return:
        """
        if tree.root.parent is not None:
            raise ValueError("Root's parent must be None.")

        for _ in range(n):
            action_node = _get_next_node(tree.state, self.tree_policy)
            action_node.reward = self.default_policy(tree)
            self.backup(node)

        return rand_max(root.children.values(), key=lambda x: x.q).action

    def _expand(state_node):
        action = np.random.choice(state_node.untried_actions)
        return state_node.children[action].sample_state()

    def _best_child(state_node, tree_policy):
        best_action_node = utils.rand_max(state_node.children.values(),
                                          key=tree_policy)
        return best_action_node.sample_state()


    def _get_next_node(state_node, tree_policy):
        while not state_node.is_terminal():
            if state_node.untried_actions == []:
                return _expand(state_node)
            else:
                state_node = _best_child(state_node, tree_policy)
        return state_node
