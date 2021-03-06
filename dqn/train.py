from dqn import *
from collections import namedtuple

VALID_ACTIONS = [-1, 0, 1]

def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)

## NOTE:
# change a decimal value to ternary representation
def dec2tern_str(n):
    if n == 0:
        return '0' * NUM_ELEVATORS
    nums = []
    while n:
        n , r = divmod(n,NUM_VALID_ACTIONS)
        nums.append(str(r))
    res = ''.join(reversed(nums))
    if len(res) == NUM_ELEVATORS:
        return res
    else:
        prefix = '0' * (NUM_ELEVATORS - len(res))
        res = ''.join([prefix, res])
        return res

# converts action vector [e1, e2, .., en] to indices in NUM_VALID_ACTION ** NUM_ELEV vector
def action_vec2index(action):
    mul = np.flip(3 ** np.arange(NUM_ELEVATORS), 0)
    aa = np.array([x+1 for x in action])
    return np.sum(aa * mul)

def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action_idx = np.argmax(q_values)
        act_per_elevator = []
        A[best_action_idx] += (1.0 - epsilon)
        return A

    return policy_fn


def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=500000,
                    replay_memory_init_size=50000,
                    update_target_estimator_every=10000,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    batch_size=32,
                    record_video_every=50):

    EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards", "episode_avg_wait"])
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        sess: Tensorflow Session object
        env: OpenAI environment
        q_estimator: Estimator object used for the q values
        target_estimator: Estimator object used for the targets
        num_episodes: Number of episodes to run for
        experiment_dir: Directory to save Tensorflow summaries in
        replay_memory_size: Size of the replay memory
        replay_memory_init_size: Number of random experiences to sample when initializing
          the reply memory.
        update_target_estimator_every: Copy parameters from the Q estimator to the
          target estimator every N steps
        discount_factor: Lambda time discount factor
        epsilon_start: Chance to sample a random action when taking an action.
          Epsilon is decayed over time and this is the start value
        epsilon_end: The final minimum value of epsilon after decaying is done
        epsilon_decay_steps: Number of steps to decay epsilon over
        batch_size: Size of batches to sample from the replay memory
        record_video_every: Record a video every N episodes

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    # The replay memory
    replay_memory = []

    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        episode_avg_wait=np.zeros(num_episodes))


    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver = tf.train.Saver()
    # Load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print "Loading model checkpoint {}...\n".format(latest_checkpoint)
        saver.restore(sess, latest_checkpoint)

    total_t = sess.run(tf.contrib.framework.get_global_step())

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = make_epsilon_greedy_policy(
        q_estimator, NUM_VALID_ACTIONS ** NUM_ELEVATORS)

    def get_action(action_probs):
        idx = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        action = [int(x)-1 for x in list(dec2tern_str(idx))]
        assert len(action) == NUM_ELEVATORS
        return action

    # Populate the replay memory with initial experience
    print "Populating replay memory..."
    state = env.reset()
    for _ in range(replay_memory_init_size):
        action_probs = policy(sess, state, epsilons[min(total_t, epsilon_decay_steps-1)])
        action = get_action(action_probs)
        next_state, reward, done = env.step(action)
        replay_memory.append(Transition(state, action, reward, next_state, done))
        if done:
            state = env.reset()
        else:
            state = next_state

    for i_episode in range(num_episodes):

        # Save the current checkpoint
        saver.save(tf.get_default_session(), checkpoint_path)

        # Reset the environment
        state = env.reset()
        dream = env
        loss = None
        """
        # NOTE create an Monte Carlo Tree
        mcstate = MCState(state, reward, env)
        mcts = MCTS(tree_policy=UCB1, default_policy=immediate_reward, backup=monte_carlo)
        """
        for t in itertools.count():

            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]


            # Add epsilon to Tensorboard
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=epsilon, tag="epsilon")
            q_estimator.summary_writer.add_summary(episode_summary, total_t)

            # Maybe update the target estimator
            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, q_estimator, target_estimator)
                #print "Copied model parameters to target network."

            # Print out which step we're on, useful for debugging.
            #print "Step {} ({}) @ Episode {}/{}, loss: {}".format(t, total_t, i_episode + 1, num_episodes, loss)
            #sys.stdout.flush()

            # Take a step
            action_probs = policy(sess, state, epsilon)
            action = get_action(action_probs)
            next_state, reward, done = env.step(action)

            # NOTE use tree policy to reach leaf node
            # then expand from leaf node
            # best_action = mcts(MCState.root)

            # If our replay memory is full, pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            # Save transition to replay memory
            replay_memory.append(Transition(state, action, reward, next_state, done))

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # Sample a minibatch from the replay memory
            samples = random.sample(replay_memory, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))


            # convert action_batch in the form of indices from batch of vectors.
            action_batch = [action_vec2index(a) for a in action_batch]

            # Calculate q values and targets (Double DQN)
            # aligned in batch sizes
            q_values_next = q_estimator.predict(sess, next_states_batch)
            best_actions = np.argmax(q_values_next, axis=1)
            q_values_next_target = target_estimator.predict(sess, next_states_batch)

            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                            discount_factor * q_values_next_target[np.arange(batch_size), best_actions]

            # Perform gradient descent update
            states_batch = np.array(states_batch)
            loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)

            if done:
                break

            state = next_state
            total_t += 1

        env.update_global_time_list()
        avg_time = sum(env.global_time_list) / float(len(env.global_time_list))
        stats.episode_avg_wait[i_episode] = avg_time

        # Add summaries to tensorboard
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=stats.episode_rewards[i_episode]/float(stats.episode_lengths[i_episode]), node_name="episode_reward", tag="episode_reward")
        episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], node_name="episode_length", tag="episode_length")
        episode_summary.value.add(simple_value=stats.episode_avg_wait[i_episode], node_name="episode_avg_wait", tag="episode_avg_wait")
        q_estimator.summary_writer.add_summary(episode_summary, total_t)
        q_estimator.summary_writer.flush()

        yield total_t, i_episode, EpisodeStats(
            episode_lengths=stats.episode_lengths[:i_episode+1],
            episode_rewards=stats.episode_rewards[:i_episode+1]/float(stats.episode_lengths[i_episode]),
            episode_avg_wait=stats.episode_avg_wait[:i_episode+1])

    return
