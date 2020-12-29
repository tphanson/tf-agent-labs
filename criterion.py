import os
from tf_agents.utils import common
import matplotlib.pyplot as plt
import ray

from agent.dqn import DQN
from env import OhmniInSpace

ray.init()


@ray.remote
def eval_single_episode(max_steps):
    # Init env & agent
    tfenv = OhmniInSpace.env()
    CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  './models/checkpoints')
    dqn = DQN(tfenv, CHECKPOINT_DIR)
    dqn.agent.train = common.function(dqn.agent.train)
    # Eval
    time_step = tfenv.reset()
    steps = max_steps
    episode_return = 0.0
    while not time_step.is_last():
        steps -= 1
        action_step = dqn.agent.policy.action(time_step)
        time_step = tfenv.step(action_step.action)
        episode_return += time_step.reward
    episode_return += time_step.reward*steps  # Amplify the return
    return episode_return.numpy()[0]


class ExpectedReturn:
    def __init__(self):
        self.returns = None
        self.max_steps = 500

    def eval_multiple_episodes(self, num_episodes):
        futures = [
            eval_single_episode.remote(self.max_steps)
            for _ in range(num_episodes)
        ]
        episode_returns = ray.get(futures)
        avg_return = sum(episode_returns) / num_episodes
        return avg_return

    def eval(self, num_episodes=5):
        avg_return = self.eval_multiple_episodes(num_episodes)
        if self.returns is None:
            self.returns = [avg_return]
        else:
            self.returns.append(avg_return)
        return avg_return

    def save(self):
        plt.plot(self.returns)
        plt.ylabel('Average Return')
        plt.savefig('models/eval.jpg')
