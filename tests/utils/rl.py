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
			'corrected_values': [],
		}
		self.long_term_attributes = {}
		self.rand = {}
		for attribute in self.attributes:
			self.long_term_attributes
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
		size = min(len(self.attributes['states_0']), size)
		samples = {}
		for _ in range(size):
			idx = np.random.choice(len(self.attributes['states_0']))
			for attribute in self.attributes:
				if attribute not in samples:
					samples[attribute] = self.attributes[attribute][idx]
				else:
					samples[attribute] = np.concatenate([samples[attribute], self.attributes[attribute][idx]], axis=0)
		return samples

	def discount_values(self, v, discount):
		self.attributes['corrected_values'] = corrected_values = []
		for i, predicted_value in enumerate(self.attributes['predicted_values_0'][::-1]):
			step = -(i + 1)
			assert step < 0, 'ERROR'
			if step == -1:
				corrected_values.append(discount * np.array(v) + self.attributes['rewards'][step][0])
			else:
				corrected_values.append(discount * np.array(corrected_values[i - 1]) + self.attributes['rewards'][step][0])
		corrected_values = corrected_values[::-1]
		self.attributes['corrected_values'] = corrected_values
		# print(v)
		# for i, value in enumerate(corrected_values):
		# 	print(value[0][0], self.attributes['rewards'][i][0], self.attributes['predicted_values_0'][i][0][0])
		# quit()
		return corrected_values












