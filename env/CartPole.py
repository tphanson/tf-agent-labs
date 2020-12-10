from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment


class TfEnv():
    def __init__(self):
        self.name = 'CartPole-v0'

    def gen_env(self):
        """ Convert pyenv to tfenv """
        pyenv = suite_gym.load(self.name)
        tfenv = tf_py_environment.TFPyEnvironment(pyenv)
        return tfenv
