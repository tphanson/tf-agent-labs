import os
import time
import tensorflow as tf
from tf_agents.utils import common

from env import OhmniInSpace
from agent.dqn import DQN
from buffer import ReplayBuffer
from eval import ExpectedReturn

# Compulsory config for tf_agents
tf.compat.v1.enable_v2_behavior()

# Trick
# No GPU: my super-extra-fast-and-furiuos-ahuhu machine
# GPUs: training servers
LOCAL = not len(tf.config.list_physical_devices('GPU')) > 0

# Saving dir
POLICY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          './models/policy')
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              './models/checkpoints')

# Environment
train_env = OhmniInSpace.env()
eval_env = OhmniInSpace.env()

# Agent
dqn = DQN(train_env, CHECKPOINT_DIR)
dqn.agent.train = common.function(dqn.agent.train)

# Metrics and Evaluation
criterion = ExpectedReturn()

# Replay buffer
initial_collect_steps = 2000
replay_buffer = ReplayBuffer(
    dqn.agent.collect_data_spec,
    batch_size=train_env.batch_size,
)
for _ in range(initial_collect_steps):
    replay_buffer.collect_steps(train_env, dqn.agent.collect_policy)
dataset = replay_buffer.as_dataset()
iterator = iter(dataset)

# Train
num_iterations = 20000
eval_step = 1000
start = time.time()
loss = 0
step = dqn.agent.train_step_counter.numpy()
while step <= num_iterations:
    if LOCAL:
        train_env.render()
    replay_buffer.collect_steps(train_env, dqn.agent.collect_policy)
    experience, _ = next(iterator)
    loss += dqn.agent.train(experience).loss
    # Evaluation
    step = dqn.agent.train_step_counter.numpy()
    if step % eval_step == 0:
        # Checkpoints
        dqn.save_checkpoint()
        # Evaluation
        avg_return = criterion.eval(eval_env, dqn.agent.policy, num_episodes=3)
        print('Step = {0}: Average Return = {1} / Average Loss = {2}'.format(
            step, avg_return, loss/eval_step))
        end = time.time()
        print('Step estimated time: {:.4f}'.format((end-start)/eval_step))
        # Reset
        start = time.time()
        loss = 0

# Visualization
criterion.save()
