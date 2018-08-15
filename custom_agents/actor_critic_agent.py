'''
An agent trained using vanilla Actor Critic
'''

import tensorflow as tf
import numpy as np

from pommerman.agents import BaseAgent
from pommerman import constants
from pommerman import utility

from ..utils.models import CNN


class ActorCriticAgent(BaseAgent):
	
	def __init__(self, name, *args, **kwargs):
		super(ActorCriticAgent, self).__init__(*args, **kwargs)
		self.name = name

	def episode_end(self, reward):
		print('{}\'s score: {}'.format(self.name, reward))

	def act(self, obs, action_space):
		# Base Network
		self.observations = tf.placeholder(
			shape=(None, 11, 11, 12), # types of values in obs['board'], compressing 10-13 to self, allies and enemies
			dtype=tf.float32,
			name='observations'
		)
		base_network = CNN(
			name='base_network',
			inputs=self.observations,
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
			output_dim=6, # number of possible actions
		)
		# Take Action
		# Store Rollout
		# Update Policies

