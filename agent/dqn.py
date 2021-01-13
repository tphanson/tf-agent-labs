import tensorflow as tf
from tensorflow import keras
from tf_agents.agents import CategoricalDqnAgent
from tf_agents.utils import common

from networks import q_rnn_agent, categorical_q_rnn_network


class DQN():
    def __init__(self, env, checkpoint_dir):
        # Env
        self.env = env
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=0.00001)
        # Policy
        self.conv = keras.Sequential([  # (96, 96, *)
            keras.layers.Conv2D(  # (92, 92, *)
                filters=32, kernel_size=(5, 5), strides=(1, 1), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),  # (46, 46, *)
            keras.layers.Conv2D(  # (42, 42, 32)
                filters=64, kernel_size=(5, 5), strides=(1, 1), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),  # (21, 21, *)
            keras.layers.Conv2D(  # (10, 10, *)
                filters=128, kernel_size=(3, 3), strides=(2, 2), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),  # (5, 5, *)
            keras.layers.Flatten(),
        ])
        self.q_net = categorical_q_rnn_network.CategoricalQRnnNetwork(
            self.env.observation_spec(),
            self.env.action_spec(),
            num_atoms=51,
            preprocessing_layers=self.conv,
            input_fc_layer_params=(768,),
            lstm_size=(512,),
            output_fc_layer_params=(512, 256),
        )
        # Agent
        self.agent = q_rnn_agent.CategoricalQRnnAgent(
            self.env.time_step_spec(),
            self.env.action_spec(),
            categorical_q_network=self.q_net,
            optimizer=self.optimizer,
            min_q_value=-3,
            max_q_value=1,
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
