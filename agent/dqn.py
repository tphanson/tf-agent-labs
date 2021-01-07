import tensorflow as tf
from tensorflow import keras
from tf_agents.agents import categorical_dqn
from tf_agents.networks import categorical_q_network
from tf_agents.utils import common
from tf_agents.experimental.train.utils import strategy_utils

GPU = len(tf.config.list_physical_devices('GPU')) > 0


class DQN():
    def __init__(self, env, checkpoint_dir):
        # Env
        self.env = env
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=0.00001)
        with strategy_utils.get_strategy(tpu=False, use_gpu=GPU).scope():
            # Policy
            self.processor = self._build_processor()
            self.q_net = categorical_q_network.CategoricalQNetwork(
                self.env.observation_spec(),
                self.env.action_spec(),
                num_atoms=51,
                preprocessing_layers=self.processor,
                fc_layer_params=(512, 256),
            )
            # Agent
            self.agent = categorical_dqn.categorical_dqn_agent.CategoricalDqnAgent(
                self.env.time_step_spec(),
                self.env.action_spec(),
                categorical_q_network=self.q_net,
                optimizer=self.optimizer,
                min_q_value=-3,
                max_q_value=1,
                n_step_update=2,
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
        # Accumulative layer (infinite-term memory)
        self.encoder = self.q_net.get_layer(index=0).get_layer(index=0)

    def _build_processor(self):
        # Input layer
        x = keras.layers.Input(shape=(96, 96, 3))
        # Convolutional layer
        conv = keras.Sequential([  # (96, 96, *)
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
            keras.layers.Dense(768, activation='relu'),
        ])
        y = conv(x)
        # # Feedback layer
        # feed = keras.Sequential([
        #     keras.layers.Dense(768, activation='relu'),
        #     keras.layers.Dense(768, activation='relu'),
        # ])
        # zeros = tf.zeros([32, 256])
        # v = feed(zeros)
        # # Combiner
        # conc = keras.layers.Concatenate()
        # y = conc([x, v])
        # Output layer
        return keras.Model(inputs=x, outputs=y)

    def call_encoder(self, inputs):
        return self.encoder(inputs)

    def save_checkpoint(self):
        self.checkpointer.save(self.global_step)

    def load_checkpoint(self):
        self.checkpointer.initialize_or_restore()
        self.global_step = tf.compat.v1.train.get_global_step()
