import os
import numpy as np
import tensorflow as tf

from env import CartPole
from agent.dqn import DQN

# Compulsory config for tf_agents
tf.compat.v1.enable_v2_behavior()

# Saving dir
POLICY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          './models/policy')
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              './models/checkpoints')

# Environment
env = CartPole.env()

# Agent
dqn = DQN(env, CHECKPOINT_DIR)
dqn.load_checkpoint()

counter = 0
while counter < 10000:
    counter += 1
    time_step = env.current_time_step()
    action_step = dqn.agent.policy.action(time_step)
    print('Action:', np.squeeze(action_step.action.numpy()))
    env.step(action_step.action)
    env.render()
