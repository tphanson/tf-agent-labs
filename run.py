import os
import tensorflow as tf

from env import OhmniInSpace
from agent.dqn import DQN

# Compulsory config for tf_agents
tf.compat.v1.enable_v2_behavior()

# Saving dir
POLICY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          './models/policy')
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              './models/checkpoints')

# Environment
env = OhmniInSpace.env(gui=True)

# Agent
dqn = DQN(env, CHECKPOINT_DIR)
dqn.load_checkpoint()

counter = 0
while counter < 10000:
    counter += 1
    time_step = env.current_time_step()
    action = dqn.agent.policy.action(time_step).action
    _, reward, _, _ = env.step(action)
    print('Action: {}, Reward: {}'.format(action.numpy(), reward.numpy()))
    env.render()
