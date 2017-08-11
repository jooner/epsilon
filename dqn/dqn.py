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
        depth, height, width, channel = self.s_dim
        self.inputs = tf.placeholder(shape=[None, depth, height, width, channel], dtype=tf.float32)
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
        # X = tf.expand_dims(tf.to_float(self.inputs),-1)
        initializer = tf.contrib.layers.xavier_initializer()
        # 1st conv layer
        kernel_1 = tf.get_variable('kernel_1', shape=[3,2,2,2,8],
                                    dtype=tf.float32, initializer=initializer)
        bias_1 = tf.get_variable('bias_1', shape=[8], dtype=tf.float32, initializer=initializer)
        conv1 = tf.nn.conv3d(self.inputs, filter=kernel_1, strides=[1,1,1,1,1], padding='SAME')
        conv1 = tf.nn.bias_add(conv1, bias_1)
        conv1 = tf.nn.relu(conv1, name='relu_1')
        pool1 = tf.nn.max_pool3d(conv1, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')
        # 2nd conv layer
        kernel_2 = tf.get_variable('kernel_2', shape=[3,2,2,8,16],
                                    dtype=tf.float32, initializer=initializer)
        bias_2 = tf.get_variable('bias_2', shape=[16], dtype=tf.float32, initializer=initializer)
        conv2 = tf.nn.conv3d(pool1, filter=kernel_2, strides=[1,1,1,1,1], padding='SAME')
        conv2 = tf.nn.bias_add(conv2, bias_2)
        conv2 = tf.nn.relu(conv2, name='relu_2')
        pool2 = tf.nn.max_pool3d(conv2, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')
        # 3rd conv layer
        kernel_3 = tf.get_variable('kernel_3', shape=[3,2,2,16,4],
                                    dtype=tf.float32, initializer=initializer)
        bias_3 = tf.get_variable('bias_3', shape=[4], dtype=tf.float32, initializer=initializer)
        conv3 = tf.nn.conv3d(pool2, filter=kernel_3, strides=[1,1,1,1,1], padding='SAME')
        conv3 = tf.nn.bias_add(conv3, bias_3)
        conv3 = tf.nn.relu(conv3, name='relu_3')

        """
        conv1 = tf.contrib.layers.conv2d(
            X, num_outputs=64, kernel_size=3, stride=2, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 32, 2, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 8, 2, 1, activation_fn=tf.nn.relu)
        """
        flattened = tf.contrib.layers.flatten(conv3)
        # Fully connected layers with RELU
        fc1 = tf.contrib.layers.fully_connected(flattened, 128)
        self.predictions = tf.contrib.layers.fully_connected(fc1, self.a_dim, activation_fn=None)
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), self.actions_pl)



        # Calcualte the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001,
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
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions)),
            tf.summary.scalar("avg_q_value", tf.reduce_mean(self.predictions))
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
