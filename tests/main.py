import numpy as np

from agent import ActorCriticAgent
from gridworld import GridWorld

env = GridWorld()
agent = ActorCriticAgent(savepath='saved_models', load=False)

for i in np.arange(1, 10001):
	obs, end, reward = env.reset()
	while end == 0:
		# obs = np.zeros((5,5,1))
		# obs = np.concatenate((obs, np.ones_like(obs)*0/200), axis=-1)
		# obs[0,0,0] = 1
		action, predicted_value = agent.act(obs, reward)
		# print(action, predicted_value)
		# quit()
		obs, end, reward = env.step(action)
	obs = np.zeros((5,5,2))
	# obs = np.concatenate((obs, np.zeros_like(obs)), axis=-1)
	# obs = np.concatenate((obs, np.ones_like(obs)*0/200), axis=-1)
	obs[1,0,0] = 1
	obs[1,1,1] = 1
	obs[2,2,1] = 1
	action, predicted_value = agent.act(obs, reward=None)
	print(predicted_value)
	# quit()
	agent.episode_end(end, reward)
	print('Total Reward: {}'.format(env.total_reward))
	