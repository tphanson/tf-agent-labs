import os
import numpy as np
import tensorflow as tf

from env import OhmniInSpace


# Saving dir
POLICY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          './models/policy')
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              './models/checkpoints')

# Environment
env = OhmniInSpace.env(gui=True)

counter = 0
while counter < 10000:
    counter += 1
    time_step = env.current_time_step()
    action = 24
    _, reward, _, _ = env.step(action)
    print('Action: {}, Reward: {}'.format(action, reward.numpy()))
    env.render()
