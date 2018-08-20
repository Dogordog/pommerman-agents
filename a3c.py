import threading
import multiprocessing
import psutil
import numpy as np
import tensorflow as tf
import scipy.signal
from time import sleep
import os
import json
from absl import flags
from absl import app

import pommerman
from pommerman import agents
import custom_agents

## HELPER FUNCTIONS

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
	from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
	to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
	op_holder = []
	for from_var,to_var in zip(from_vars,to_vars):
		op_holder.append(to_var.assign(from_var))
	return op_holder

def process_observation(obs, agent):

	board = obs['board']
	position = obs['position']
	bomb_life = obs['bomb_life']
	bomb_blast_strength = obs['bomb_blast_strength']
	enemies = obs['enemies']

	state = None
	for i in np.arange(10):
		if state is None:
			state = np.expand_dims((board == i).astype(np.float32), axis=-1)
		else:
			state = np.concatenate([state, np.expand_dims((board == i).astype(np.float32), axis=-1)], axis=-1)
	
	enemy_positions = None
	valid_ids = [10, 11, 12, 13]
	valid_ids.pop(agent.agent_id) # self.agent_id goes from 0 to 3
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

	if agent.can_kick == True:
		can_kick = np.ones_like(self_position)
	else:
		can_kick = np.zeros_like(self_position)
	state = np.concatenate([state, can_kick], axis=-1)

	ammo = np.ones_like(self_position) * agent.ammo
	state = np.concatenate([state, ammo], axis=-1)

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

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
	return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
	def _initializer(shape, dtype=None, partition_info=None):
		out = np.random.randn(*shape).astype(np.float32)
		out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
		return tf.constant(out)
	return _initializer

# Sample from distribution of arguments
def sample_dist(dist):
	return np.random.choice(np.arange(len(dist[0])), p=dist[0])

## ACTOR-CRITIC NETWORK

class AC_Network():
	def __init__(self, scope, trainer):
		with tf.variable_scope(scope):
			# Architecture here follows Atari-net Agent described in [1] Section 4.3
			channels = 16

			self.inputs = tf.placeholder(shape=[None,11,11,channels], dtype=tf.float32)
			conv_1 = tf.layers.conv2d(
				inputs=self.inputs,
				filters=16,
				kernel_size=[2,2],
				strides=[1,1],
				padding='same',
				activation=tf.nn.relu,
				name='conv_1'
			)
			conv_2 = tf.layers.conv2d(
				inputs=conv_1,
				filters=32,
				kernel_size=[2,2],
				strides=[1,1],
				padding='same',
				activation=tf.nn.relu,
				name='conv_2',
			)
			unroll = tf.reshape(conv_2, [-1, 11*11*32])
			dense = tf.layers.dense(
				inputs=unroll,
				units=256,
				activation=None,
				name='dense',
			)
			self.latent_vector = dense
			
			# Output layers for policy and value estimations
			self.policy = tf.layers.dense(
				inputs=self.latent_vector,
				units=6,
				activation=tf.nn.softmax,
				kernel_initializer=normalized_columns_initializer(0.01),
			)
			self.value = tf.layers.dense(
				inputs=self.latent_vector,
				units=1,
				kernel_initializer=normalized_columns_initializer(1.0),
			)
			# Only the worker network need ops for loss functions and gradient updating.
			# calculates the losses
			# self.gradients - gradients of loss wrt local_vars
			# applies the gradients to update the global network
			if scope != 'global':
				self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
				self.actions_onehot = tf.one_hot(self.actions, 6, dtype=tf.float32)
				self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
				self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

				self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
				
				# Loss functions
				self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))

				self.log_policy = tf.log(tf.clip_by_value(self.policy, 1e-20, 1.0)) # avoid NaN with clipping when value in policy becomes zero
				self.entropy_loss = - tf.reduce_sum(self.policy * self.log_policy)

				self.policy_loss = - tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_outputs, 1e-20, 1.0)) * self.advantages)

				self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy_loss * 0.01

				# Get gradients from local network using local losses
				local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
				self.gradients = tf.gradients(self.loss,local_vars)
				self.var_norms = tf.global_norm(local_vars)
				grads, self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
				
				# Apply local gradients to global network
				global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
				self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))

## WORKER AGENT

class Worker():
	def __init__(self, name, trainer, model_path, global_episodes, global_steps):
		self.name = "worker_" + str(name)
		self.number = name	
		self.model_path = model_path
		self.trainer = trainer
		self.global_episodes = global_episodes
		self.increment_global_episodes = self.global_episodes.assign_add(1)
		self.global_steps = global_steps
		self.increment_global_steps = self.global_steps.assign_add(1)
		self.episode_rewards = []
		self.episode_lengths = []
		self.episode_mean_values = []
		self.summary_writer = tf.summary.FileWriter(model_path + "/train_" + str(self.number))

		#Create the local copy of the network and the tensorflow op to copy global paramters to local network
		self.local_AC = AC_Network(self.name, trainer)
		self.update_local_ops = update_target_graph('global', self.name)  
		
		print('Initializing environment #{}...'.format(self.number))
		agent_list = [
			custom_agents.StaticAgent(),
			agents.SimpleAgent(),
			agents.SimpleAgent(),
			agents.SimpleAgent(),
		]
		self.env = pommerman.make('PommeFFACompetition-v0', agent_list)
		self.agent = agent_list[0]

		
	def train(self, rollout, sess, gamma, bootstrap_value):
		rollout = np.array(rollout)
		obs = rollout[:,0]
		actions = rollout[:,1]
		rewards = rollout[:,2]
		next_obs = rollout[:,3]
		values = rollout[:,5]

		# Here we take the rewards and values from the rollout, and use them to calculate the advantage and discounted returns. 
		# The advantage function uses generalized advantage estimation from [2]
		self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
		discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
		self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
		advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
		advantages = discount(advantages,gamma)

		# Update the global network using gradients from loss
		# Generate network statistics to periodically save
		feed_dict = {self.local_AC.target_v:discounted_rewards,
			self.local_AC.inputs:np.stack(obs).reshape(-1,11,11,16),
			self.local_AC.actions:actions,
			self.local_AC.advantages:advantages}
		v_l,p_l,e_l,g_n,v_n, _ = sess.run([self.local_AC.value_loss,
			self.local_AC.policy_loss,
			self.local_AC.entropy_loss,
			self.local_AC.grad_norms,
			self.local_AC.var_norms,
			self.local_AC.apply_grads],
			feed_dict=feed_dict)
		return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n,v_n
		
	def work(self,max_episode_length,gamma,sess,coord,saver):
		episode_count = sess.run(self.global_episodes)
		total_steps = 0
		print ("Starting worker " + str(self.number))
		with sess.as_default(), sess.graph.as_default():				 
			while not coord.should_stop():
				# Download copy of parameters from global network
				sess.run(self.update_local_ops)

				episode_buffer = []
				episode_values = []
				episode_frames = []
				episode_reward = 0
				episode_step_count = 0
				
				# Start new episode
				obs = self.env.reset()
				episode_frames.append(obs[0])
				state = process_observation(obs[0], self.agent)
				s = state
				prev_wood_wall = np.sum(obs[0]['board'] == 2)
				episode_end = False
				
				while not episode_end and 10 in obs[0]['alive']:
					
					# Take an action using distributions from policy networks' outputs.
					action_dist, v = sess.run(
						[
							self.local_AC.policy, 
							self.local_AC.value,
						],
						feed_dict={
							self.local_AC.inputs: state,
						}
					)
						
					action = sample_dist(action_dist)
					
					if FLAGS.render:
						self.env.render()
					actions = self.env.act(obs)
					obs, r, episode_end, info = self.env.step([action] + actions[1:])
					wood_wall = np.sum(obs[0]['board'] == 2)
					
					if self.agent.prev_enemies != (len(obs[0]['alive']) - 1) and 10 in obs[0]['alive']:
						r = 10.
					elif self.agent.prev_ammo == (self.agent.ammo - 1) and prev_wood_wall != wood_wall: # need to differentiate between ammo decreasing, ammo returning to me and ammo increasing due to powerup
						r = 1.
					else:
						r = 0.
					prev_wood_wall = wood_wall

					state = process_observation(obs[0], self.agent)

					if not episode_end:
						episode_frames.append(obs[0])
						s1 = state
					else:
						s1 = s
					
					# Append latest state to buffer
					episode_buffer.append([s,action,r,s1,episode_end,v[0,0]])
					episode_values.append(v[0,0])

					episode_reward += r
					s = s1				 
					sess.run(self.increment_global_steps)
					total_steps += 1
					episode_step_count += 1
					
					# If the episode hasn't ended, but the experience buffer is full, then we make an update step using that experience rollout.
					if len(episode_buffer) == 30 and not episode_end and episode_step_count != max_episode_length - 1:
						# Since we don't know what the true final return is, we "bootstrap" from our current value estimation.
						v1 = sess.run(self.local_AC.value, 
							feed_dict={self.local_AC.inputs: state})[0,0]
						v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,v1)
						episode_buffer = []
						sess.run(self.update_local_ops)
					if episode_end:
						break
											
				self.episode_rewards.append(episode_reward)
				self.episode_lengths.append(episode_step_count)
				self.episode_mean_values.append(np.mean(episode_values))
				episode_count += 1

				global _max_score, _running_avg_score
				if _max_score < episode_reward:
					_max_score = episode_reward
				_running_avg_score = (2.0 / 101) * (episode_reward - _running_avg_score) + _running_avg_score

				if episode_count % 10 == 0:
					print("{} Step #{} Episode #{} Reward: {}".format(self.name, total_steps, episode_count, episode_reward))
					print("Total Steps: {}\tTotal Episodes: {}\tMax Score: {}\tAvg Score: {:.3f}".format(sess.run(self.global_steps), sess.run(self.global_episodes), _max_score, _running_avg_score))

				# Update the network using the episode buffer at the end of the episode.
				if len(episode_buffer) != 0:
					v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.0)

				if episode_count % 50 == 0 and episode_count != 0:
					if episode_count % 250 == 0 and self.name == 'worker_0':
						saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
						print ("Saved Model")

					mean_reward = np.mean(self.episode_rewards[-50:])
					mean_length = np.mean(self.episode_lengths[-50:])
					mean_value = np.mean(self.episode_mean_values[-50:])
					summary = tf.Summary()
					summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
					summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
					summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
					summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
					summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
					summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
					summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
					summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
					self.summary_writer.add_summary(summary, episode_count)

					self.summary_writer.flush()
				
				sess.run(self.increment_global_episodes)

def main(unused_args):
	max_episode_length = 300
	gamma = .99 # discount rate for advantage estimation and reward discounting
	
	global _max_score, _running_avg_score
	_max_score = 0
	_running_avg_score = 0

	tf.reset_default_graph()

	model_path = 'saved_models'
	tf.gfile.MakeDirs(model_path)

	with tf.device("/cpu:0"): 
		global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
		global_steps = tf.Variable(0, dtype=tf.int32, name='global_steps', trainable=False)
		trainer = tf.train.AdamOptimizer(learning_rate=1e-3)
		master_network = AC_Network('global', None) # Generate global network
		if FLAGS.n_agents < 1:
			num_workers = psutil.cpu_count() # Set workers to number of available CPU threads
		else:
			num_workers = FLAGS.n_agents
		workers = []
		# Create worker classes
		for i in range(num_workers):
			workers.append(Worker(i, trainer, model_path, global_episodes, global_steps))
		saver = tf.train.Saver(max_to_keep=5)

	with tf.Session() as sess:
		coord = tf.train.Coordinator()
		if FLAGS.load:
			print ('Loading Model...')
			ckpt = tf.train.get_checkpoint_state(model_path)
			saver.restore(sess, ckpt.model_checkpoint_path)
		else:
			sess.run(tf.global_variables_initializer())

		if FLAGS.render:
			# Rendering doesn't seem to work in threads
			workers[0].work(max_episode_length, gamma, sess, coord, saver)
		else:
			# This is where the asynchronous magic happens.
			# Start the "work" process for each worker in a separate thread.
			worker_threads = []
			for worker in workers:
				worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
				t = threading.Thread(target=(worker_work))
				t.start()
				sleep(0.5)
				worker_threads.append(t)
			coord.join(worker_threads)

if __name__ == '__main__':
	flags.DEFINE_bool('load', False, 'Load')
	flags.DEFINE_bool('render', False, 'Render')
	flags.DEFINE_integer('n_agents', 4, 'Number of agents')
	FLAGS = flags.FLAGS
	app.run(main)
