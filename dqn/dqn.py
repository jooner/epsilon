import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf

"""
Orignal Code provided by https://github.com/dennybritz/reinforcement-learning/blob/master/DQN/dqn.py

"""

if "../" not in sys.path:
  sys.path.append("../")

from epsilon.globals import *
from collections import deque, namedtuple

class Estimator():
    """
    Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, state_dim, action_dim, batch_size=32, scope="estimator", summaries_dir=None):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.scope = scope
        self.batch_size = batch_size
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """
        # Placeholders for our input
        self.inputs = tf.placeholder(shape=[None, self.s_dim], dtype=tf.float32)
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
        # Fully connected layers with RELU
        fc1 = tf.contrib.layers.fully_connected(self.inputs, 512)
        fc2 = tf.contrib.layers.fully_connected(fc1, 512)
        fc3 = tf.contrib.layers.fully_connected(fc2, 128, activation_fn=None)
        self.predictions = tf.contrib.layers.fully_connected(fc3, self.a_dim, activation_fn=None)
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), self.actions_pl)

        # Calcualte the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00025,
                                                   decay=0.99,
                                                   momentum=0.0,
                                                   epsilon=1e-6)
        self.train_op = self.optimizer.minimize(self.loss,
                                                global_step=tf.contrib.framework.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])


    def predict(self, sess, s):
        """
        Predicts action values.

        Args:
          sess: Tensorflow session

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated
          action values.
        """
        return sess.run(self.predictions, { self.inputs: s })

    def update(self, sess, s, a, y):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 4, 160, 160, 3]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        feed_dict = { self.inputs: s, self.y_pl: y, self.actions_pl: a }
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss
