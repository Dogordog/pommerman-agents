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
	
	def __init__(self, name, load=False, savepath=None, *args, **kwargs):
		super(ActorCriticAgent, self).__init__(*args, **kwargs)
		self.name = name
		self.savepath = savepath
		self.reset_memory()
		self.update_every = 16
		self.discount = 0.9
		self.sess = tf.Session()
		self.wins = 0
		self.losses = 0
		with tf.variable_scope(self.name):
			self.build_model()
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=3)
		self.sess.run(tf.global_variables_initializer())
		if load and self.savepath is not None:
			self.load(self.sess, self.savepath, verbose=True)

	def reset_memory(self):
		self.steps = 0
		self.memory = Memory()
		self.prev_state = np.zeros((1, 11, 11, 16))
		self.prev_value = [[0]]
		self.prev_action = 0
		self.prev_ammo = None

	def episode_end(self, reward):
		if reward == 1:
			self.wins += 1
		else:
			self.losses += 1
		# Store rollout with episode reward
		self.memory.store(
			self.prev_state,
			self.prev_state,
			self.prev_value,
			[[0]],
			reward,
			self.prev_action
		)
		loss, v_loss, e_loss, p_loss = self.update_networks()
		if self.savepath is not None:
			self.save(self.sess, self.savepath)
		# print('Step #{} - Loss: {:.3f}'.format(self.steps, loss))
		# print('Value Loss: {:.3f} - Entropy Loss: {:.3f} - Policy Loss: {:.3f}'.format(v_loss, e_loss, p_loss))
		print('End of Episode - {}\'s score: {}'.format(self.name, reward))
		print('Wins: {} - Losses: {} - Win Ratio: {:.3f}'.format(self.wins, self.losses, self.wins / (self.wins + self.losses)))
		self.reset_memory()

	def act(self, obs, action_space):
		if self.prev_ammo is None:
			self.prev_ammo = self.ammo
			self.prev_blast_strength = self.blast_strength
			self.prev_can_kick = self.can_kick
		self.steps += 1
		# Store rollout
		#	reward = 0 until end of game
		#	might consider -0.1 instead
		state = self.process_observation(obs['board'], obs['position'], obs['bomb_life'], obs['bomb_blast_strength'], obs['enemies'])
		feed_dict = {
			self.states: state,
		}
		action_dist, predicted_value = self.sess.run([self.predicted_actions, self.value_network.outputs], feed_dict)
		
		# Calculate reward
		# Add subreward for collecting powerups
		if self.prev_ammo != self.ammo: # need to differentiate between ammo decreasing, ammo returning to me and ammo increasing due to powerup
			print('AMMO CHANGED')
			# reward = .6
		elif self.prev_blast_strength != self.blast_strength:
			print('STRENGTH CHANGED')
			# reward = 5.
		elif self.prev_can_kick != self.can_kick:
			print('KICK CHANGED')
			# reward = 5.
		# else:
		reward = -.1
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
			loss, v_loss, e_loss, p_loss = self.update_networks()
			self.memory.reset()
			# print('Step #{} - Loss: {:.3f}'.format(self.steps, loss))
			# print('Value Loss: {:.3f} - Entropy Loss: {:.3f} - Policy Loss: {:.3f}'.format(v_loss, e_loss, p_loss))
		# Take action
		action = sample_dist(action_dist[0])

		self.prev_state = state
		self.prev_action = action
		self.prev_value = predicted_value
		self.prev_ammo = self.ammo
		self.prev_blast_strength = self.blast_strength
		self.prev_can_kick = self.can_kick
		
		return action

	def build_model(self):
		self.states = tf.placeholder(
			shape=(None, 11, 11, 16), # types of values in obs['board'], compressing 10-13 to self, allies and enemies
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
			output_dim=128,
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

		self.actions_taken_onehot = tf.one_hot(self.actions_taken, ACTIONS_SIZE, dtype=tf.float32)

		# Total loss = value_loss + entropy_loss + policy_loss
		self.value_loss = value_loss = tf.losses.mean_squared_error(labels=self.bootstrapped_values, predictions=self.value_network.outputs)
		self.entropy_loss = entropy_loss = -tf.reduce_sum(tf.log(tf.clip_by_value(self.predicted_actions, 1e-20, 1.0)) * self.predicted_actions)
		self.policy_loss = policy_loss = -tf.reduce_sum(tf.log(tf.clip_by_value(tf.reduce_sum(self.predicted_actions * self.actions_taken_onehot, axis=1), 1e-20, 1.0)) * self.advantages)
		self.loss = 0.5 * value_loss - 0.01 * entropy_loss + policy_loss
		self.optimize = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

	def update_networks(self):
		samples = self.memory.sample(8)
		corrected_values = self.discount * np.array(samples['predicted_values_1']) + np.array(samples['rewards'])
		advantages = np.array(corrected_values) - np.array(samples['predicted_values_0'])
		feed_dict = {
			self.states: np.array(samples['states_0']),
			self.bootstrapped_values: np.array(samples['predicted_values_1']) - np.array(samples['rewards']),
			self.actions_taken: np.array(samples['actions'], dtype=np.int32),
			self.advantages: advantages,
		}
		loss, v_loss, e_loss, p_loss, _ = self.sess.run([self.loss, self.value_loss, self.entropy_loss, self.policy_loss, self.optimize], feed_dict)
		return loss, v_loss, e_loss, p_loss

	def process_observation(self, board, position, bomb_life, bomb_blast_strength, enemies):
		state = None
		for i in np.arange(10):
			if state is None:
				state = np.expand_dims((board == i).astype(np.float32), axis=-1)
			else:
				state = np.concatenate([state, np.expand_dims((board == i).astype(np.float32), axis=-1)], axis=-1)
		
		enemy_positions = None
		valid_ids = [10, 11, 12, 13]
		valid_ids.pop(self.agent_id) # self.agent_id goes from 0 to 3
		for enemy in enemies:
			enemy_id = enemy.value
			if enemy_id in valid_ids:
				valid_ids.pop(valid_ids.index(enemy_id))
			if enemy_positions is None:
				enemy_positions = np.expand_dims((board == enemy_id).astype(np.float32), axis=-1)
			else:
				enemy_positions += np.expand_dims((board == enemy_id).astype(np.float32), axis=-1)
		state = np.concatenate([state, enemy_positions], axis=-1)

		# No ally for FFA
		# ally_id = valid_ids[0]
		# ally_position = np.expand_dims((board == ally_id).astype(np.float32), axis=-1)
		# state = np.concatenate([state, ally_position], axis=-1)

		self_position = np.zeros_like(enemy_positions)
		self_position[position[0], position[1]] = 1
		state = np.concatenate([state, self_position], axis=-1)

		if self.can_kick == True:
			can_kick = np.ones_like(self_position)
		else:
			can_kick = np.zeros_like(self_position)
		state = np.concatenate([state, can_kick], axis=-1)

		ammo = np.ones_like(self_position) * self.ammo
		state = np.concatenate([state, ammo], axis=-1)

		state = np.concatenate([state, np.expand_dims(bomb_life, axis=-1)], axis=-1)
		state = np.concatenate([state, np.expand_dims(bomb_blast_strength, axis=-1)], axis=-1)

		return np.expand_dims(state, axis=0)

	def save(self, sess, savepath, global_step=None, prefix="ckpt", verbose=False):
		if savepath[-1] != '/':
			savepath += '/'
		self.saver.save(sess, savepath + prefix, global_step=global_step)
		if verbose:
			print("Model saved to {}.".format(savepath + prefix + '-' + str(global_step)))

	def load(self, sess, savepath, verbose=False):
		if savepath[-1] != '/':
			savepath += '/'
		ckpt = tf.train.latest_checkpoint(savepath)
		self.saver.restore(sess, ckpt)
		if verbose:
			print("Model loaded from {}.".format(ckpt))