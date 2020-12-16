import tensorflow as tf
from tensorflow import keras
from tf_agents.agents import dqn
from tf_agents.networks import q_network
from tf_agents.utils import common


class DQN():
    def __init__(self, env, checkpoint_dir):
        # Env
        self.env = env
        # Policy
        self.preprocessing_layers = keras.Sequential([  # (96, 96, *)
            keras.layers.Conv2D(  # (92, 92, 16)
                filters=16, kernel_size=(5, 5), strides=(1, 1), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),  # (46, 46, 16)
            keras.layers.Conv2D(  # (42, 42, 32)
                filters=32, kernel_size=(5, 5), strides=(1, 1), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),  # (21, 21, 32)
            keras.layers.Conv2D(  # (10, 10, 64)
                filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),  # (5, 5, 64)
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
        ])
        self.q_net = q_network.QNetwork(
            self.env.observation_spec(),
            self.env.action_spec(),
            preprocessing_layers=self.preprocessing_layers,
            fc_layer_params=(128, 32))
        # Agent
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=0.00000075)
        self.agent = dqn.dqn_agent.DqnAgent(
            self.env.time_step_spec(),
            self.env.action_spec(),
            q_network=self.q_net,
            optimizer=self.optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=self.global_step)
        self.agent.initialize()
        # Checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.checkpointer = common.Checkpointer(
            ckpt_dir=self.checkpoint_dir,
            max_to_keep=1,
            agent=self.agent,
            policy=self.agent.policy,
            global_step=self.global_step
        )

    def save_checkpoint(self):
        self.checkpointer.save(self.global_step)

    def load_checkpoint(self):
        self.checkpointer.initialize_or_restore()
        self.global_step = tf.compat.v1.train.get_global_step()
