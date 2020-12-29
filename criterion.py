import matplotlib.pyplot as plt


class ExpectedReturn:
    def __init__(self):
        self.returns = None
        self.max_steps = 500
    
    def eval_single_episode(self, tfenv, agent):
        time_step = tfenv.reset()
        steps = self.max_steps
        episode_return = 0.0
        while not time_step.is_last():
            steps -= 1
            action_step = agent.action(time_step)
            time_step = tfenv.step(action_step.action)
            episode_return += time_step.reward.numpy()[0]
        episode_return += time_step.reward*steps  # Amplify the return
        return episode_return

    def eval_multiple_episodes(self, tfenv, agent, num_episodes):
        total_returns = 0.0
        for _ in range(num_episodes):
            total_returns += self.eval_single_episode(tfenv, agent)
        avg_return = total_returns / num_episodes
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
