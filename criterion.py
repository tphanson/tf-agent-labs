import matplotlib.pyplot as plt
from multiprocessing import Pool


def eval_single_episode(max_steps, tfenv, agent):
    tfenv = gen_tfenv_func()
    time_step = tfenv.reset()
    steps = max_steps
    episode_return = 0.0
    while not time_step.is_last():
        steps -= 1
        action_step = agent.action(time_step)
        time_step = tfenv.step(action_step.action)
        episode_return += time_step.reward.numpy()[0]
    episode_return += time_step.reward*steps  # Amplify the return
    return episode_return


class ExpectedReturn:
    def __init__(self):
        self.returns = None
        self.max_steps = 500

    def eval_multiple_episodes(self, gen_tfenv_func, agent, num_episodes):
        pool = Pool()
        args = []
        for _ in range(num_episodes):
            args.append((self.max_steps, gen_tfenv_func, agent))
        print(args)
        episode_returns = pool.map(eval_single_episode, args)
        print(episode_returns)
        avg_return = sum(episode_returns) / num_episodes
        return avg_return

    def eval(self, gen_tfenv_func, agent, num_episodes=5):
        avg_return = self.eval_multiple_episodes(
            gen_tfenv_func, agent, num_episodes)
        if self.returns is None:
            self.returns = [avg_return]
        else:
            self.returns.append(avg_return)
        return avg_return

    def save(self):
        plt.plot(self.returns)
        plt.ylabel('Average Return')
        plt.savefig('models/eval.jpg')
