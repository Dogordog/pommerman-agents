import numpy as np

class GridWorld():

	def __init__(self):
		self.board = np.zeros((5, 5, 1))
		self.good_goal = (4, 4)
		self.candy_1 = (1, 1)
		self.candy_1_eaten = False
		self.candy_2 = (2, 2)
		self.candy_2_eaten = False
		self.bad_goal = (4, 0)
		self.actions = np.arange(4)
		self.time = 0
		self.total_reward = 0

	def reset(self):
		self.candy_1_eaten = False
		self.candy_2_eaten = False
		# self.agent_position = [np.random.choice(5), np.random.choice(5)]
		self.agent_position = [0, 0]
		while self.in_goal(self.agent_position) != 0:
			 self.agent_position = [np.random.choice(5), np.random.choice(5)]
		self.set_agent_position()
		self.total_reward = self.check_reward(self.agent_position)
		self.time = 0
		return self.board, self.in_goal(self.agent_position), 0

	def step(self, action):
		self.time += 1
		self.update_agent_position(action)
		reward = self.check_reward(self.agent_position)
		self.total_reward += reward
		if self.time == 200:
			return self.board, 1, reward
		return self.board, self.in_goal(self.agent_position), reward

	def update_agent_position(self, action):
		if action == 0: # UP
			self.agent_position[1] += 1
		elif action == 1: # LEFT
			self.agent_position[0] += -1
		elif action == 2: # DOWN
			self.agent_position[1] += -1
		elif action == 3: # RIGHT
			self.agent_position[0] += 1
		for i in [0, 1]:
			if self.agent_position[i] < 0:
				self.agent_position[i] = 0
			elif self.agent_position[i] >= 5:
				self.agent_position[i] = 4
		self.set_agent_position()

	def set_agent_position(self):
		self.board = np.zeros((5, 5, 1))
		self.board[self.agent_position[0], self.agent_position[1], 0] = 1

		candy_board = np.zeros_like(self.board)
		if not self.candy_1_eaten:
			candy_board[self.candy_1[0], self.candy_1[1], 0] = 1
		if not self.candy_2_eaten:
			candy_board[self.candy_1[0], self.candy_2[1], 0] = 1
		self.board = np.concatenate((self.board, candy_board), axis=-1)

		# self.board = np.concatenate((self.board, np.ones_like(candy_board)*self.time/200), axis=-1)

	def check_reward(self, position):
		if position[0] == self.candy_1[0] and position[1] == self.candy_1[1] and not self.candy_1_eaten:
			self.candy_1_eaten = True
			return 3
		if position[0] == self.candy_2[0] and position[1] == self.candy_2[1] and not self.candy_2_eaten:
			self.candy_2_eaten = True
			return 3
		return 0

	def in_goal(self, position):
		if position[0] == self.good_goal[0] and position[1] == self.good_goal[1]:
			return 1
		if position[0] == self.bad_goal[0] and position[1] == self.bad_goal[1]:
			return -1
		return 0
