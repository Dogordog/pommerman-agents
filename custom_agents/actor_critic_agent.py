'''
An agent trained using vanilla Actor Critic
'''

import tensorflow as tf
import numpy as np

from pommerman.agents import BaseAgent
from pommerman import constants
from pommerman import utility

from ..utils.models import CNN
from ..utils.rl import Memory
from ..utils import sample_dist


ACTIONS_SIZE = len(constants.Action())

class ActorCriticAgent(BaseAgent):
	
	def __init__(self, name, *args, **kwargs):
		super(ActorCriticAgent, self).__init__(*args, **kwargs)
		self.name = name
		self.reset()
		self.update_every = 10
		with tf.variable_scope(self.name):
			self.build_model()

	def reset(self):
		self.steps = 0
		self.memory = Memory()
		self.prev_state = np.zeros((11, 11, 12))
		self.prev_value = 0
		self.prev_action = 0

	def episode_end(self, reward):
		# Store rollout with episode reward
		self.memory.store(
			self.prev_state,
			self.prev_state,
			self.prev_value,
			self.prev_value,
			reward,
			self.prev_action
		)
		print('{}\'s score: {}'.format(self.name, reward))
		self.reset()

	def act(self, obs, action_space):
		self.steps += 1
		# Store rollout
		#	reward = 0 until end of game
		#	might consider -0.1 instead
		state = obs['board']
		feed_dict = {
			self.state: obs['board'],
		}
		action_dist, predicted_value = self.sess.run([self.policy_network.outputs, self.value_network.outputs], feed_dict)
		reward = 0
		self.memory.store(
			self.prev_state,
			state,
			self.prev_value,
			predicted_value,
			reward,
			self.prev_action
		)
		# Update networks
		if self.steps % self.update_every == 0:
			loss = self.update_networks()
			print('Step #{} - Loss: {:.3f}'.format(self.steps, loss))
		# Take action
		action = sample_dist(action_dist)

		self.prev_state = state
		self.prev_action = action
		self.prev_value = predicted_value

		return action_space.sample()

	def build_model(self):
		self.states = tf.placeholder(
			shape=(None, 11, 11, 12), # types of values in obs['board'], compressing 10-13 to self, allies and enemies
			dtype=tf.float32,
			name='state'
		)
		self.bootstrapped_values = tf.placeholder(
			shape=(None),
			dtype=tf.float32,
			name='bootstrapped_values',
		)
		self.action_update = tf.placeholder(
			shape=(None, ACTIONS_SIZE)
		)
		# Base Network
		base_network = CNN(
			name='base_network',
			inputs=self.states,
			output_dim=64,
		)
		# Value Network
		value_network = FFN(
			name='value_network',
			inputs=base_network.outputs,
			output_dim=1,
		)
		# Policy Network
		policy_network = FFN(
			name='value_network',
			inputs=base_network.outputs,
			output_dim=ACTIONS_SIZE, # number of possible actions
		)

		# Total loss = value_loss + entropy_loss + policy_loss
		value_loss = tf.losses.mean_squared_error(labels=self.bootstrapped_value, predictions=self.value_network.outputs)
		entropy_loss = -tf.reduce_sum(tf.log(policy_network.outputs))
		policy_loss = tf.losses.mean_squared_error(labels=self.action_update, predictions=self.policy_network.outputs)
		self.loss = value_loss + entropy_loss + policy_loss
		self.optimize = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

	def update_networks(self):
		samples = self.memory.sample(8)
		corrected_values = samples['predicted_values_1'] - samples['rewards']
		advantage = corrected_values - samples['predicted_values_0']
		feed_dict = {
			self.states: samples['states_0'],
			self.bootstrapped_values: samples['predicted_values_1'] - samples['rewards'],
			self.action_update: samples['actions'] * advantage
		}
		loss, _ = self.sess.run([self.loss, self.optimize], feed_dict)
		return loss



