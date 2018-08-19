import tensorflow as tf
import numpy as np

from utils.models import CNN, FFN
from utils.rl import Memory
from utils import sample_dist


ACTIONS_SIZE = 4

class ActorCriticAgent():
	
	def __init__(self, savepath, load=False):
		self.name = 'agent'
		self.savepath = savepath
		self.reset_memory()
		self.update_every = 30
		self.discount = 0.99
		self.sess = tf.Session()
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model()
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=3)
		self.sess.run(tf.global_variables_initializer())
		if self.savepath is not None:
			tf.gfile.MakeDirs('/'.join([self.savepath]))
		if load and self.savepath is not None:
			self.load(self.sess, '/'.join([self.savepath]), verbose=True)

	def reset_memory(self):
		self.steps = 0
		self.memory = Memory()
		self.prev_state = np.zeros((1, 5, 5, 2))
		self.prev_value = [[0]]
		self.prev_action = 0
		self.prev_ammo = None

	def episode_end(self, end, reward):
		# Store rollout with episode reward
		self.memory.store(
			self.prev_state,
			self.prev_state,
			self.prev_value,
			[[0]],
			end+reward,
			self.prev_action
		)
		loss, v_loss, e_loss, p_loss = self.update_networks(v=[[0]])
		if self.savepath is not None:
			self.save(self.sess, self.savepath)
		if end == -1:
			print('End of Episode: LOSS')
		elif end == 1:
			print('End of Episode: WON')
		else:
			print('ERROR')
		self.reset_memory()

	def act(self, obs, reward=None):
		self.steps += 1
		# Store rollout
		#	reward = 0 until end of game
		#	might consider -0.1 instead
		state = np.expand_dims(obs, axis=0)
		feed_dict = {
			self.states: state,
		}
		action_dist, predicted_value = self.sess.run([self.predicted_actions, self.value_network.outputs], feed_dict)
		
		# Calculate reward
		# Add subreward for collecting powerups
		# reward = 0
		
		if reward is not None:
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
				loss, v_loss, e_loss, p_loss = self.update_networks(v=predicted_value)
				self.memory.reset()
				# print('Step #{} - Loss: {:.3f}'.format(self.steps, loss))
				# print('Value Loss: {:.3f} - Entropy Loss: {:.3f} - Policy Loss: {:.3f}'.format(v_loss, e_loss, p_loss))
			# Take action
		action = sample_dist(action_dist[0])

		self.prev_state = state
		self.prev_action = action
		self.prev_value = predicted_value
		
		return action, predicted_value

	def build_model(self):
		self.states = tf.placeholder(
			shape=(None, 5, 5, 2), # types of values in obs['board'], compressing 10-13 to self, allies and enemies
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

		self.actions_taken_onehot = tf.one_hot(self.actions_taken, ACTIONS_SIZE, dtype=tf.float32)

		# Total loss = value_loss + entropy_loss + policy_loss
		self.value_loss = value_loss = tf.losses.mean_squared_error(labels=self.bootstrapped_values, predictions=self.value_network.outputs)
		self.entropy_loss = entropy_loss = -tf.reduce_sum(tf.log(tf.clip_by_value(self.predicted_actions, 1e-20, 1.0)) * self.predicted_actions)
		self.policy_loss = policy_loss = -tf.reduce_sum(tf.log(tf.clip_by_value(tf.reduce_sum(self.predicted_actions * self.actions_taken_onehot, axis=1), 1e-20, 1.0)) * self.advantages)
		self.loss = 0.5 * value_loss - 0.01 * entropy_loss + policy_loss
		self.optimize = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

	def update_networks(self, v):
		self.memory.discount_values(v, discount=self.discount)
		samples = self.memory.sample(16)
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
