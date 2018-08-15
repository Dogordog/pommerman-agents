import numpy as np


class Memory():

	def __init__(self):
		self.attributes = {
			'states_0': [],
			'states_1': [],
			'predicted_values_0': [],
			'predicted_values_1': [],
			'rewards': [],
			'actions': [],
		}
		self.rand = {}
		for attribute in self.attributes:
			self.rand[attribute] = np.random.RandomState(42)

	def reset(self):
		for attribute in self.attributes:
			self.attributes[attribute] = []

	def store(self, state_0, state_1, predicted_value_0, predicted_value_1, reward, action):
		self.attributes['states_0'].append(state_0)
		self.attributes['states_1'].append(state_1)
		self.attributes['predicted_values_0'].append(predicted_value_0)
		self.attributes['predicted_values_1'].append(predicted_value_1)
		self.attributes['rewards'].append([reward])
		self.attributes['actions'].append([action])

	def shuffle(self):
		for attribute in self.attributes:
			self.rand[attribute].shuffle(self.attributes[attribute])

	def sample(self, size, shuffle=True):
		if shuffle:
			self.shuffle()
		samples = {}
		for _ in range(size):
			idx = np.random.choice(len(self.attributes['states_0']))
			for attribute in self.attributes:
				if attribute not in samples:
					samples[attribute] = self.attributes[attribute][idx]
				else:
					samples[attribute] = np.concatenate([samples[attribute], self.attributes[attribute][idx]], axis=0)
		return samples

