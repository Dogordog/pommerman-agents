import numpy as np

class GridWorld():

	def __init__(self):
		self.board = np.zeros((5, 5, 1))
		self.good_goal = (4, 4)
		self.bad_goal = (4, 0)
		self.actions = np.arange(4)

	def reset(self):
		self.agent_position = [np.random.choice(5), np.random.choice(5)]
		while self.in_goal(self.agent_position) != 0:
			 self.agent_position = [np.random.choice(5), np.random.choice(5)]
		self.set_agent_position()
		return self.board, self.in_goal(self.agent_position)

	def step(self, action):
		self.update_agent_position(action)
		return self.board, self.in_goal(self.agent_position)

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

	def in_goal(self, position):
		if position[0] == self.good_goal[0] and position[1] == self.good_goal[1]:
			return 1
		if position[0] == self.bad_goal[0] and position[1] == self.bad_goal[1]:
			return -1
		return 0
