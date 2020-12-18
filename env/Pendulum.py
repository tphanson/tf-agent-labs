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
        self.input_shape = self.image_shape+(3,)
        self._env = gym.make('Pendulum-v0')
        # Discretized actions
        self._num_actions = 16
        self._action_ref = np.linspace(
            self._env.action_space.low[0],
            self._env.action_space.high[0],
            self._num_actions
        )
        # Env specs
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32,
            minimum=0, maximum=self._num_actions-1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=self.input_shape, dtype=np.float32,
            minimum=self._env.observation_space.low,
            maximum=self._env.observation_space.high,
            name='observation')
        self._state = None
        self._episode_ended = False
        self._observation = None
        # Reset
        self._reset()

    def __nomarlize(self, img):
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        (h, w) = img.shape
        img = img[int(h*0.2): int(h*0.8), int(w*0.2): int(w*0.8)]
        img = cv.resize(img, self.image_shape)
        img = np.reshape(img, self.image_shape+(1,))
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
        if self._state is None:
            init_state = observation
            (_, _, stack_channel) = self.input_shape
            for _ in range(stack_channel-1):
                init_state = np.append(init_state, observation, axis=2)
            self._state = np.array(init_state, dtype=np.float32)
        self._state = self._state[:, :, 1:]
        self._state = np.append(self._state, observation, axis=2)

    def get_info(self):
        return {}

    def _reset(self):
        self._env.reset()
        self._episode_ended = False
        self._state = None
        self.set_state()
        return ts.restart(self._state)

    def _step(self, action):
        action = [self._action_ref[action]]
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

        drawed_img = np.copy(img)
        drawed_img = cv.resize(drawed_img, (512, 512))
        cv.imshow('CartPole-v1', drawed_img)
        cv.waitKey(10)

        return img


def env():
    """ Convert pyenv to tfenv """
    pyenv = PyEnv()
    tfenv = tf_py_environment.TFPyEnvironment(pyenv)
    return tfenv
