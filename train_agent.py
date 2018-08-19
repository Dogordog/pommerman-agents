# Train agent on replays

import pickle
import numpy as np
import custom_agents

agent = custom_agents.ActorCriticAgent(name='smith', update_turn=0, savepath='saved_models', load=False)

replays = []
for i in np.arange(4):
	with open('replays/data_{}.pkl'.format(i), 'rb') as file:
		replays += pickle.load(file).replays

batchsize = 256

for i in np.arange(1, 501):
	np.random.shuffle(replays)
	for replay in replays:
		size = len(replay.attributes['states_0'])
		steps = np.random.choice(np.arange(size), batchsize)
		state_0 = []
		state_1 = []
		reward = []
		action = []
		value = []
		for j in steps:
			state_0.append(replay.attributes['states_0'][j])
			state_1.append(replay.attributes['states_1'][j])
			reward.append(replay.attributes['rewards'][j])
			action.append(replay.attributes['actions'][j])
			value.append(replay.attributes['values'][j])

		feed_dict = {
			agent.replay_actions: np.eye(6)[np.array(action)].reshape(batchsize, 6),
			agent.bootstrapped_values: np.array(value).reshape(batchsize),
			agent.states: np.array(state_0).reshape(batchsize, 11, 11, 16),
		}
		replay_loss, v_loss, a_loss, a_accuracy, _ = agent.sess.run([agent.replay_loss, agent.value_loss, agent.replay_action_loss, agent.replay_action_accuracy, agent.replay_optimize], feed_dict)
	if i % 5 == 0:
		print('Training Epoch #{}'.format(i))
		print('Replay Loss: {:.3f} - Value Loss: {:.3f} - Action Loss: {:.3f}'.format(replay_loss, v_loss, a_loss))
		print('Accuracy: {:.3f}'.format(a_accuracy))
		agent.save(agent.sess, '/'.join(['saved_models', agent.name]), verbose=True)