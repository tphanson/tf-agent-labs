import pybullet as p
import pybullet_data
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2 as cv

from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from env.objs import floor, ohmni, obstacle

VELOCITY_COEFFICIENT = 6
INTERPRETER = [[-1, -1], [-0.5, 0.5], [0, 0], [0.5, -0.5], [1, 1]]


class Env:
    def __init__(self, gui=False, num_of_obstacles=4, image_shape=(96, 96)):
        self.gui = gui
        self.timestep = 0.05
        self.num_of_obstacles = num_of_obstacles
        self.image_shape = image_shape
        self.clientId = self._init_ws()
        self._left_wheel_id = 0
        self._right_wheel_id = 1

    def _init_ws(self):
        """
        Create server and start, there are two modes:
        1. GUI: it visualizes the environment and allow controlling
            ohmni via sliders.
        2. Headless: by running everything in background, it's suitable
            for ai/ml/rl development.
        """
        # Init server
        clientId = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=clientId)
        p.setTimeStep(self.timestep, physicsClientId=clientId)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # Return
        return clientId

    def _randomize_destination(self):
        destination = (np.random.rand(2)*20-10).astype(dtype=np.float32)
        p.addUserDebugLine(
            np.append(destination, 0.),  # From
            np.append(destination, 3.),  # To
            [1, 0, 0],  # Red
            physicsClientId=self.clientId
        )
        return destination

    def _build(self):
        """ Including floor, ohmni, obstacles into the environment """
        # Add gravity
        p.setGravity(0, 0, -10, physicsClientId=self.clientId)
        # Add plane and ohmni
        floor(self.clientId, texture=False, wall=False)
        ohmniId, _capture_image = ohmni(self.clientId)
        # Add obstacles at random positions
        for _ in range(self.num_of_obstacles):
            obstacle(self.clientId)
        # Return
        return ohmniId, _capture_image

    def _reset(self):
        """ Remove all objects, then rebuild them """
        p.resetSimulation(physicsClientId=self.clientId)
        self.ohmniId, self._capture_image = self._build()
        self.destination = self._randomize_destination()

    def capture_image(self):
        """ Get image from navigation camera """
        if self._capture_image is None:
            raise ValueError('_capture_image is undefined')
        return self._capture_image(self.image_shape)

    def getContactPoints(self):
        """ Get Ohmni contacts """
        return p.getContactPoints(self.ohmniId, physicsClientId=self.clientId)

    def getBasePositionAndOrientation(self):
        """ Get Ohmni position and orientation """
        return p.getBasePositionAndOrientation(self.ohmniId, physicsClientId=self.clientId)

    def reset(self):
        """ Reset the environment """
        self._reset()

    def step(self, action):
        """ Controllers for left/right wheels which are separate """
        # Normalize velocities
        [left_wheel, right_wheel] = INTERPRETER[action]
        left_wheel = left_wheel*VELOCITY_COEFFICIENT
        right_wheel = right_wheel*VELOCITY_COEFFICIENT
        # Step
        p.setJointMotorControl2(self.ohmniId, self._left_wheel_id,
                                p.VELOCITY_CONTROL,
                                targetVelocity=left_wheel,
                                physicsClientId=self.clientId)
        p.setJointMotorControl2(self.ohmniId, self._right_wheel_id,
                                p.VELOCITY_CONTROL,
                                targetVelocity=right_wheel,
                                physicsClientId=self.clientId)
        p.stepSimulation(physicsClientId=self.clientId)


class PyEnv(py_environment.PyEnvironment):
    def __init__(self, gui=False, image_shape=(96, 96)):
        super(PyEnv, self).__init__()
        # Parameters
        self.image_shape = image_shape
        self.input_shape = self.image_shape + (3,)
        self._num_of_obstacles = 0
        self._max_steps = 500
        # PyEnvironment variables
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32,  minimum=0, maximum=4, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=self.input_shape, dtype=np.float32,
            minimum=0, maximum=1, name='observation')
        # Init bullet server
        self._env = Env(
            gui,
            num_of_obstacles=self._num_of_obstacles,
            image_shape=self.image_shape
        )
        # Internal states
        self._state = None
        self._episode_ended = False
        self._num_steps = 0
        # Reset
        self._reset()

    def _get_image_state(self):
        _, _, rgb_img, _, seg_img = self._env.capture_image()
        img = np.array(rgb_img, dtype=np.float32)/255
        mask = np.minimum(seg_img, 1, dtype=np.float32)
        mask = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
        return img, mask

    def _get_pose_state(self):
        position, orientation = self._env.getBasePositionAndOrientation()
        position = np.array(position, dtype=np.float32)
        destination_posistion = np.append(self._env.destination, 0.)
        rotation = R.from_quat(
            [-orientation[0], -orientation[1], -orientation[2], orientation[3]])
        rel_position = rotation.apply(destination_posistion - position)
        _pose = rel_position[0:2]
        return _pose.astype(dtype=np.float32)

    def _normalized_distance_to_destination(self):
        """ Compute the distance from agent to destination """
        position, _ = self._env.getBasePositionAndOrientation()
        position = np.array(position[0:2], dtype=np.float32)
        distance = np.linalg.norm(position-self._env.destination)
        origin = np.linalg.norm(np.zeros([2])-self._env.destination)
        return min(distance/origin, 1)

    def _is_fatal(self):
        """ Compute whether there are collisions or not """
        position, orientation = self._env.getBasePositionAndOrientation()
        position = np.array(position, dtype=np.float32)
        collision = self._env.getContactPoints()
        for contact in collision:
            # Contact with things different from floor
            if contact[2] != 0:
                return True
        # Ohmni felt out of the environment
        if abs(position[2]) >= 0.5:
            return True
        # Ohmni is falling down
        if abs(orientation[0]) > 0.2 or abs(orientation[1]) > 0.2:
            return True
        return False

    def _compute_reward(self):
        """ Compute reward and return (<stopped>, <reward>) """
        normalized_distance = self._normalized_distance_to_destination()
        # Ohmni reach the destination
        if normalized_distance < 0.1:
            return True, self._max_steps - self._num_steps
        # Stop if detecting collisions or a fall
        if self._is_fatal():
            return True, 0
        # Ohmni on his way
        return False, 1

    def _reset(self):
        """ Reset environment"""
        self._env.reset()
        self._state = None
        self._episode_ended = False
        self._num_steps = 0
        self.set_state()
        return ts.restart(self._state)

    def action_spec(self):
        """ Return action specs """
        return self._action_spec

    def observation_spec(self):
        """ Return observation specs """
        return self._observation_spec

    def get_info(self):
        return {}

    def get_state(self):
        return self._state

    def set_state(self, _unused=None):
        # Gamifying
        (h, w) = self.image_shape
        _, mask = self._get_image_state()  # Image state
        pose = self._get_pose_state()  # Pose state
        cent = np.array([w/2, h/2], dtype=np.float32)
        dest = -pose*1000 + cent  # Transpose/Scale/Tranform
        mask = cv.line(mask,
                       (int(cent[1]), int(cent[0])),
                       (int(dest[1]), int(dest[0])),
                       (0, 1, 0), thickness=2)
        observation = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)
        observation = np.reshape(observation, self.image_shape+(1,))
        # Set state
        if self._state is None:
            init_state = observation
            (_, _, stack_channel) = self.input_shape
            for _ in range(stack_channel-1):
                init_state = np.append(init_state, observation, axis=2)
            self._state = np.array(init_state, dtype=np.float32)
        self._state = self._state[:, :, 1:]
        self._state = np.append(self._state, observation, axis=2)

    def _step(self, action):
        """ Step, action is velocities of left/right wheel """
        # Reset if ended
        if self._episode_ended:
            return self.reset()
        self._num_steps += 1
        # Step the environment
        self._env.step(action)
        # Compute and save states
        self.set_state()
        self._episode_ended, reward = self._compute_reward()
        # If exceed the limitation of steps, return rewards
        if self._num_steps > self._max_steps:
            self._episode_ended = True
            return ts.termination(self._state, reward)
        # Transition
        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward)

    def render(self, mode='rgb_array'):
        """ Show video stream from navigation camera """
        img = self.get_state()

        drawed_img = np.copy(img)
        drawed_img = cv.cvtColor(drawed_img, cv.COLOR_RGB2BGR)
        drawed_img = cv.resize(drawed_img, (512, 512))
        cv.imshow('OhmniInSpace-v0', drawed_img)
        cv.waitKey(10)

        return img


def env(gui=False):
    """ Convert pyenv to tfenv """
    pyenv = PyEnv(gui=gui)
    tfenv = tf_py_environment.TFPyEnvironment(pyenv)
    return tfenv