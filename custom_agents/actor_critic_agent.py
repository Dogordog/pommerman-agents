'''
An agent trained using vanilla Actor Critic
'''

import tensorflow as tf
import numpy as np

from pommerman.agents import BaseAgent
from pommerman import constants
from pommerman import utility

from utils.models import CNN, FFN
from utils.rl import Memory
from utils import sample_dist


ACTIONS_SIZE = 6

class ActorCriticAgent(BaseAgent):
	
	def __init__(self, name, *args, **kwargs):
		super(ActorCriticAgent, self).__init__(*args, **kwargs)
		self.name = name
		self.reset_memory()
		self.update_every = 10
		self.discount = 0.9
		self.sess = tf.Session()
		with tf.variable_scope(self.name):
			self.build_model()
		self.sess.run(tf.global_variables_initializer())

	def reset_memory(self):
		self.steps = 0
		self.memory = Memory()
		self.prev_state = np.zeros((1, 11, 11, 12))
		self.prev_value = [[0]]
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
		loss = self.update_networks()
		print('Step #{} - Loss: {:.3f}'.format(self.steps, loss))
		print('End of Episode - {}\'s score: {}'.format(self.name, reward))
		self.reset_memory()

	def act(self, obs, action_space):
		self.steps += 1
		# Store rollout
		#	reward = 0 until end of game
		#	might consider -0.1 instead
		state = self.process_observation(obs['board'], obs['enemies'])
		feed_dict = {
			self.states: state,
		}
		action_dist, predicted_value = self.sess.run([self.predicted_actions, self.value_network.outputs], feed_dict)
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
			self.memory.reset()
			print('Step #{} - Loss: {:.3f}'.format(self.steps, loss))
		# Take action
		action = sample_dist(action_dist[0])

		self.prev_state = state
		self.prev_action = action
		self.prev_value = predicted_value

		return action

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
		self.advantages = tf.placeholder(
			shape=(None),
			dtype=tf.float32,
		)
		self.actions_taken = tf.placeholder(
			shape=(None),
			dtype=tf.int32,
		)
		# Base Network
		self.base_network = CNN(
			name='base_network',
			inputs=self.states,
			output_dim=64,
			parent=self.name,
		)
		# Value Network
		self.value_network = FFN(
			name='value_network',
			inputs=self.base_network.outputs,
			output_dim=1,
			parent=self.name,
		)
		# Policy Network
		self.policy_network = FFN(
			name='policy_network',
			inputs=self.base_network.outputs,
			output_dim=ACTIONS_SIZE, # number of possible actions
			parent=self.name,
		)
		self.predicted_actions = tf.nn.softmax(self.policy_network.outputs)

		# Total loss = value_loss + entropy_loss + policy_loss
		value_loss = tf.losses.mean_squared_error(labels=self.bootstrapped_values, predictions=self.value_network.outputs)
		entropy_loss = -tf.reduce_sum(tf.log(tf.clip_by_value(self.policy_network.outputs, 1e-20, 1.0)) * self.policy_network.outputs)
		policy_loss = -tf.reduce_sum(tf.log(tf.clip_by_value(tf.reduce_sum(self.policy_network.outputs * tf.one_hot(self.actions_taken, ACTIONS_SIZE, dtype=tf.float32), axis=1), 1e-20, 1.0)) * self.advantages)
		self.loss = 0.5 * value_loss + 0.01 * entropy_loss + policy_loss
		self.optimize = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

	def update_networks(self):
		samples = self.memory.sample(8)
		corrected_values = np.array(samples['predicted_values_1']) - self.discount * np.array(samples['rewards'])
		advantages = np.array(corrected_values) - np.array(samples['predicted_values_0'])
		feed_dict = {
			self.states: np.array(samples['states_0']),
			self.bootstrapped_values: np.array(samples['predicted_values_1']) - np.array(samples['rewards']),
			self.actions_taken: np.array(samples['actions'], dtype=np.int32),
			self.advantages: advantages,
		}
		loss, _ = self.sess.run([self.loss, self.optimize], feed_dict)
		return loss

	def process_observation(self, board, enemies):
		state = None
		for i in np.arange(10):
			if state is None:
				state = np.expand_dims((board == i).astype(np.float32), axis=-1)
			else:
				state = np.concatenate([state, np.expand_dims((board == i).astype(np.float32), axis=-1)], axis=-1)
		
		enemy_positions = None
		valid_ids = [10, 11, 12, 13]
		for enemy in enemies:
			enemy_id = enemy.value
			if enemy_id in valid_ids:
				valid_ids.pop(valid_ids.index(enemy_id))
			if enemy_positions is None:
				enemy_positions = np.expand_dims((board == enemy_id).astype(np.float32), axis=-1)
			else:
				enemy_positions += np.expand_dims((board == enemy_id).astype(np.float32), axis=-1)
		state = np.concatenate([state, enemy_positions], axis=-1)

		ally_id = valid_ids[0]
		ally_position = np.expand_dims((board == ally_id).astype(np.float32), axis=-1)
		state = np.concatenate([state, ally_position], axis=-1)
		return np.expand_dims(state, axis=0)

