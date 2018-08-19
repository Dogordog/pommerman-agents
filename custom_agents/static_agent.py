'''The base simple agent use to train agents.
This agent is also the benchmark for other agents.
'''
from pommerman.agents import BaseAgent

class StaticAgent(BaseAgent):
    """This is a baseline agent. After you can beat it, submit your agent to
    compete.
    """

    def __init__(self, *args, **kwargs):
        super(StaticAgent, self).__init__(*args, **kwargs)
        self.prev_ammo = 1

    def episode_end(self, reward):
        """This is called at the end of the episode to let the agent know that
        the episode has ended and what is the reward.

        Args:
          reward: The single reward scalar to this agent.
        """
        self.prev_ammo = 1
        pass

    def act(self, obs, action_space):
        self.prev_ammo = self.ammo
        return 0
