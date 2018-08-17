import numpy as np

from agent import ActorCriticAgent
from gridworld import GridWorld

env = GridWorld()
agent = ActorCriticAgent(savepath='saved_models', load=True)

for i in np.arange(1000):
	obs, reward = env.reset()
	while reward == 0:
		obs = np.zeros((5,5,1))
		obs[3,4,0] = 1
		action, predicted_value = agent.act(obs)
		print(action, predicted_value)
		quit()
		obs, reward = env.step(action)
	agent.episode_end(reward)