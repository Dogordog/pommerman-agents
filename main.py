'''An example to show how to set up an pommerman game programmatically'''
from absl import flags
from absl import app

import pommerman
from pommerman import agents

import custom_agents


FLAGS = flags.FLAGS

flags.DEFINE_bool('load', False, 'Load agent')
flags.DEFINE_bool('render', False, 'Render environment')


def main(unused_args):
    '''Simple function to bootstrap a game.
       
       Use this as an example to set up your training env.
    '''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    # Create a set of agents (exactly four)
    agent_list = [
        agents.PlayerAgent(agent_control="arrows"),
        # custom_agents.ActorCriticAgent(name='smith', update_turn=0, savepath='saved_models', load=FLAGS.load, load_replay=False),
        # custom_agents.ActorCriticAgent(name='smith', update_turn=1, savepath='saved_models', load=FLAGS.load),
        # custom_agents.ActorCriticAgent(name='smith', update_turn=2, savepath='saved_models', load=FLAGS.load),
        # custom_agents.ActorCriticAgent(name='smith', update_turn=3, savepath='saved_models', load=FLAGS.load),
        # custom_agents.ActorCriticAgent(name='john', savepath='saved_models', load=FLAGS.load),
        # custom_agents.ActorCriticAgent(name='romanov', savepath='saved_models', load=FLAGS.load),
        # custom_agents.ActorCriticAgent(name='witch', savepath='saved_models', load=FLAGS.load),
        custom_agents.StaticAgent(),
        custom_agents.StaticAgent(),
        custom_agents.StaticAgent(),
        # custom_agents.DebugAgent(),
        # agents.SimpleAgent(),
        # agents.SimpleAgent(),
        # agents.SimpleAgent(),
        # agents.RandomAgent(),
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)
    # env = pommerman.make('PommeTeamCompetition-v0', agent_list)

    # Run the episodes just like OpenAI Gym
    for i_episode in range(int(1e6)):
        state = env.reset()
        done = False
        while not done:
            if FLAGS.render:
                env.render()
            actions = env.act(state)
            # print(actions)
            # quit()
            state, reward, done, info = env.step(actions)
            # if len(state[0]['alive']) < 4:
            #     reward = agent_list[0].episode_end(-1)
            #     break
            # input()
        # reward = agent_list[0].prev_sub_reward
        # avg_reward = agent_list[0].avg_sub_reward
        # if i_episode % 5 == 0:
        #     print('Episode {} finished - Reward: {:.3f} - Avg: {:.3f}'.format(i_episode, reward, avg_reward))
            # print(info)
    env.close()


if __name__ == '__main__':
    app.run(main)
