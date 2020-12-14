import gym
import numpy as np
import cv2 as cv
from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts
from env.display import virtual_display


class PyEnv(py_environment.PyEnvironment):
    def __init__(self, image_shape=(96, 96)):
        super(PyEnv, self).__init__()
        # Create env
        self.image_shape = image_shape
        self.input_shape = self.image_shape+(1,)
        self._env = gym.make('CartPole-v1')
        # Env specs
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32,
            minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=self.input_shape, dtype=np.float32,
            minimum=0, maximum=1, name='observation')
        self._state = None
        self._episode_ended = False
        # Reset
        self._reset()

    def __nomarlize(self, img):
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        (h, w) = img.shape
        img = img[int(h*0.4): int(h*0.8), :]
        img = cv.resize(img, self.image_shape)
        img = np.reshape(img, self.input_shape)
        img = np.array(img/255, dtype=np.float32)
        return img

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def get_state(self):
        return self._state

    @virtual_display
    def set_state(self, _unused=None):
        img = self._env.render(mode='rgb_array')
        observation = self.__nomarlize(img)
        self._state = observation

    def get_info(self):
        return None

    def _reset(self):
        _ = self._env.reset()
        self.set_state()
        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()
        _, reward, done, _ = self._env.step(action)
        self.set_state()
        self._episode_ended = done
        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward)

    def render(self, mode='rgb_array'):
        img = self.get_state()
        cv.imshow('CartPole-v1', img)
        cv.waitKey(10)
        return img


def env():
    """ Convert pyenv to tfenv """
    pyenv = PyEnv()
    tfenv = tf_py_environment.TFPyEnvironment(pyenv)
    return tfenv
