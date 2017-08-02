from dqn import *
from epsilon.globals import *
from epsilon.building import *
from epsilon.environment import *
import tensorflow as tf
import datetime, time
from train import *

building = Building()
env = Environment(building)

tf.reset_default_graph()

# Where we save our checkpoints and graphs
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
experiment_dir = os.path.abspath("./experiments/{}".format(st))

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

s_dim = len(env.get_state())
a_dim = NUM_ELEVATORS # 3 possible actions per elevator

# Create estimators
q_estimator = Estimator(s_dim, a_dim, scope="q", summaries_dir=experiment_dir)
target_estimator = Estimator(s_dim, a_dim, scope="target_q")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t, num_ep, stats, avg_t in deep_q_learning(sess,
                                    env,
                                    q_estimator=q_estimator,
                                    target_estimator=target_estimator,
                                    experiment_dir=experiment_dir,
                                    num_episodes=500,
                                    replay_memory_size=1000,
                                    replay_memory_init_size=1000,
                                    update_target_estimator_every=200,
                                    epsilon_start=1.0,
                                    epsilon_end=0.1,
                                    epsilon_decay_steps=500000,
                                    discount_factor=0.99,
                                    batch_size=32):

        print "Episode {}\t  Reward: {}\t Episode AvgWaitTime: {}".format(num_ep, stats.episode_rewards[-1], avg_t)
