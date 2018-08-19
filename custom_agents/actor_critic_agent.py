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
	
	def __init__(self, name, update_turn, load=False, load_replay=True, savepath=None, *args, **kwargs):
		super(ActorCriticAgent, self).__init__(*args, **kwargs)
		self.name = name
		self.update_turn = update_turn
		self.savepath = savepath
		self.memory = None
		self.reset_memory()
		self.update_every = 50
		self.discount = 0.99
		self.sess = tf.Session()
		self.wins = 0
		self.losses = 0
		self.episodes = 0
		self.sub_reward = 0
		self.prev_sub_reward = 0
		self.avg_sub_reward = 0.
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model()
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=3)
		self.sess.run(tf.global_variables_initializer())
		if self.savepath is not None:
			tf.gfile.MakeDirs('/'.join([self.savepath, self.name]))
		if load_replay and not load:
			self.load(self.sess, 'saved_models/smith_replay_trained', verbose=True)
		elif load and self.savepath is not None:
			self.load(self.sess, '/'.join([self.savepath, self.name]), verbose=True)

	def reset_memory(self):
		if self.memory is None:
			self.memory = Memory()
		else:
			self.memory.reset()
		self.steps = 0
		self.prev_state = np.zeros((1, 11, 11, 10))
		self.prev_value = [[0]]
		self.prev_action = 0
		self.prev_ammo = None
		self.prev_wood_wall = 0

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
			0,
			self.prev_action
		)
		# if ((self.episodes - self.update_turn) % 4) == 0:
		loss, v_loss, e_loss, p_loss = self.update_networks(v=[[0]])
		# if self.savepath is not None and (self.episodes - self.update_turn) % 4 == 0:
		if self.savepath is not None:
			self.save(self.sess, '/'.join([self.savepath, self.name]))
		# print('Step #{} - Loss: {:.3f}'.format(self.steps, loss))
		# print('Value Loss: {:.3f} - Entropy Loss: {:.3f} - Policy Loss: {:.3f}'.format(v_loss, e_loss, p_loss))
		# print('End of Episode - {}{}\'s score: {}'.format(self.name, self.agent_id, reward))
		# print('Wins: {} - Losses: {} - Win Ratio: {:.3f}'.format(self.wins, self.losses, self.wins / (self.wins + self.losses)))
		self.reset_memory()
		self.episodes += 1
		self.prev_sub_reward = self.sub_reward
		self.sub_reward = 0
		self.avg_sub_reward = 0.1 * self.prev_sub_reward + 0.9 * self.avg_sub_reward
		return self.avg_sub_reward

	def act(self, obs, action_space):
		if self.prev_ammo is None:
			self.prev_ammo = self.ammo
			self.prev_blast_strength = self.blast_strength
			self.prev_can_kick = self.can_kick
			self.prev_wood_wall = np.sum(obs['board'] == 2)
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
		# Add subreward for collecting powerups and destroying wooden blocks
		if self.prev_ammo == (self.ammo - 1) and self.prev_wood_wall != np.sum(obs['board'] == 2): # need to differentiate between ammo decreasing, ammo returning to me and ammo increasing due to powerup
			# print('WOOD DESTROYED')
			# print('AMMO CHANGED')
			reward = 1.
		elif self.prev_blast_strength != self.blast_strength:
			# print('STRENGTH CHANGED')
			reward = 1.
		elif self.prev_can_kick != self.can_kick:
			# print('KICK CHANGED')
			reward = 1.
		# elif :
		# 	print('WOOD DESTROYED')
		# 	reward = 1.
		else:
			# reward = -.01
			reward = 0
		self.sub_reward += reward
		
		self.memory.store(
			self.prev_state,
			state,
			self.prev_value,
			predicted_value,
			reward,
			self.prev_action
		)
		# Update networks
		# if ((self.steps % self.update_every) == 0) and (((self.episodes - self.update_turn) % 4) == 0):
		if ((self.steps % self.update_every) == 0):
			loss, v_loss, e_loss, p_loss = self.update_networks(v=predicted_value)
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
		self.prev_wood_wall = np.sum(obs['board'] == 2)
		
		return action

	def build_model(self):
		self.replay_actions = tf.placeholder(
			shape=(None, ACTIONS_SIZE),
			dtype=tf.float32,
			name='replay_actions',
		)
		self.states = tf.placeholder(
			shape=(None, 11, 11, 10), # types of values in obs['board'], compressing 10-13 to self, allies and enemies
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
			output_dim=512,
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
		self.loss = 0.5 * value_loss - 0.1 * entropy_loss + policy_loss
		self.optimize = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

		self.replay_action_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.replay_actions, logits=self.policy_network.outputs))
		self.replay_loss = self.value_loss + self.replay_action_loss
		self.replay_optimize = tf.train.AdamOptimizer(1e-4).minimize(self.replay_loss)
		self.replay_action_accuracy = tf.contrib.metrics.accuracy(labels=tf.argmax(self.replay_actions, axis=1), predictions=tf.argmax(self.policy_network.outputs, axis=1))

	def update_networks(self, v):
		self.memory.discount_values(v, discount=self.discount)
		samples = self.memory.sample(32)
		# corrected_values = self.discount * np.array(samples['predicted_values_1']) + np.array(samples['rewards'])
		advantages = np.array(samples['corrected_values']) - np.array(samples['predicted_values_0'])
		feed_dict = {
			self.states: np.array(samples['states_0']),
			self.bootstrapped_values: np.array(samples['corrected_values']),
			# self.bootstrapped_values: np.array(samples['predicted_values_1']) - np.array(samples['rewards']),
			self.actions_taken: np.array(samples['actions'], dtype=np.int32),
			self.advantages: advantages,
		}
		loss, v_loss, e_loss, p_loss, _ = self.sess.run([self.loss, self.value_loss, self.entropy_loss, self.policy_loss, self.optimize], feed_dict)
		return loss, v_loss, e_loss, p_loss

	def process_observation(self, board, position, bomb_life, bomb_blast_strength, enemies):
		state = None
		for i in [0, 1, 2, 3, 6, 7, 8]:
			if state is None:
				state = np.expand_dims((board == i).astype(np.float32), axis=-1)
			else:
				state = np.concatenate([state, np.expand_dims((board == i).astype(np.float32), axis=-1)], axis=-1)
		
		# enemy_positions = None
		# valid_ids = [10, 11, 12, 13]
		# valid_ids.pop(self.agent_id) # self.agent_id goes from 0 to 3
		# for enemy in enemies:
		# 	enemy_id = enemy.value
		# 	if enemy_id in valid_ids:
		# 		valid_ids.pop(valid_ids.index(enemy_id))
		# 	if enemy_positions is None:
		# 		enemy_positions = np.expand_dims((board == enemy_id).astype(np.float32), axis=-1)
		# 	else:
		# 		enemy_positions += np.expand_dims((board == enemy_id).astype(np.float32), axis=-1)
		# state = np.concatenate([state, enemy_positions], axis=-1)

		# No ally for FFA
		# ally_id = valid_ids[0]
		# ally_position = np.expand_dims((board == ally_id).astype(np.float32), axis=-1)
		# state = np.concatenate([state, ally_position], axis=-1)

		self_position = np.zeros((11, 11, 1))
		self_position[position[0], position[1], 0] = 1
		state = np.concatenate([state, self_position], axis=-1)

		# if self.can_kick == True:
		# 	can_kick = np.ones_like(self_position)
		# else:
		# 	can_kick = np.zeros_like(self_position)
		# state = np.concatenate([state, can_kick], axis=-1)

		# ammo = np.ones_like(self_position) * self.ammo
		# state = np.concatenate([state, ammo], axis=-1)

		state = np.concatenate([state, np.expand_dims(bomb_life, axis=-1)], axis=-1)

		bomb_blast_board = np.zeros_like(self_position)
		bomb_x, bomb_y = np.where(bomb_blast_strength != 0)
		for i, _ in enumerate(bomb_x):
			strength = bomb_blast_strength[bomb_x[i], bomb_y[i]]
			for j in np.arange(strength):
				if bomb_x[i] + j < 11:
					bomb_blast_board[int(bomb_x[i] + j), int(bomb_y[i]), 0] = 1
				if bomb_y[i] + j < 11:
					bomb_blast_board[int(bomb_x[i]), int(bomb_y[i] + j), 0] = 1
				if bomb_x[i] - j >= 0:
					bomb_blast_board[int(bomb_x[i] - j), int(bomb_y[i]), 0] = 1
				if bomb_y[i] - j >= 0:
					bomb_blast_board[int(bomb_x[i]), int(bomb_y[i] - j), 0] = 1
		state = np.concatenate([state, bomb_blast_board], axis=-1)
		
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
