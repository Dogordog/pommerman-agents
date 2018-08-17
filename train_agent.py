# Train agent on replays

import pickle
import numpy as np
import custom_agents

agent = custom_agents.ActorCriticAgent(name='smith', update_turn=0, savepath='saved_models', load=False)

with open('replays/data.pkl', 'rb') as file:
	replays = pickle.load(file).replays

batchsize = 256

for i in np.arange(1, 1001):
	np.random.shuffle(replays)
	for replay in replays:
		size = len(replay.attributes['states_0'])
		steps = np.random.choice(np.arange(size), batchsize)
		state_0 = []
		state_1 = []
		reward = []
		action = []
		value = []
		for i in steps:
			state_0.append(replay.attributes['states_0'][i])
			state_1.append(replay.attributes['states_1'][i])
			reward.append(replay.attributes['rewards'][i])
			action.append(replay.attributes['actions'][i])
			value.append(replay.attributes['values'][i])

		feed_dict = {
			agent.replay_actions: np.eye(6)[np.array(action)].reshape(batchsize, 6),
			agent.bootstrapped_values: np.array(value).reshape(batchsize),
			agent.states: np.array(state_0).reshape(batchsize, 11, 11, 16),
		}
		replay_loss, v_loss, a_loss, _ = agent.sess.run([agent.replay_loss, agent.value_loss, agent.replay_action_loss, agent.replay_optimize], feed_dict)
		if i % 50 == 0:
			print('Replay Loss: {:.3f} - Value Loss: {:.3f} - Action Loss: {:.3f}'.format(replay_loss, v_loss, a_loss))
			agent.save(agent.sess, '/'.join(['saved_models', agent.name]), verbose=True)