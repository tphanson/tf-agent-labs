import gin
import tensorflow as tf
from tf_agents.networks import network, q_rnn_network
from tf_agents.specs import tensor_spec


@gin.configurable
class CategoricalQRnnNetwork(network.Network):
    def __init__(self,
                 input_tensor_spec,
                 action_spec,
                 num_atoms=51,
                 preprocessing_layers=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 input_fc_layer_params=(75, 40),
                 lstm_size=None,
                 output_fc_layer_params=(75, 40),
                 activation_fn=tf.keras.activations.relu,
                 rnn_construction_fn=None,
                 rnn_construction_kwargs=None,
                 name='CategoricalQRnnNetwork'):

        super(CategoricalQRnnNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        if not isinstance(action_spec, tensor_spec.BoundedTensorSpec):
            raise TypeError('action_spec must be a BoundedTensorSpec. Got: %s' % (
                action_spec,))

        self._num_actions = action_spec.maximum - action_spec.minimum + 1
        self._num_atoms = num_atoms

        q_network_action_spec = tensor_spec.BoundedTensorSpec(
            (), tf.int32, minimum=0, maximum=self._num_actions * num_atoms - 1)

        self._q_network = q_rnn_network.QRnnNetwork(
            input_tensor_spec=input_tensor_spec,
            action_spec=q_network_action_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            input_fc_layer_params=input_fc_layer_params,
            lstm_size=lstm_size,
            output_fc_layer_params=output_fc_layer_params,
            activation_fn=activation_fn,
            rnn_construction_fn=rnn_construction_fn,
            rnn_construction_kwargs=rnn_construction_kwargs,
            dtype=tf.float32,
            name=name)
        self._state_spec = self._q_network.state_spec

    @property
    def num_atoms(self):
        return self._num_atoms

    def get_initial_state(self):
        return self._q_network.get_initial_state()

    def call(self, observation, step_type=None, network_state=(), training=False):
        logits, network_state = self._q_network(
            observation, step_type, network_state, training=training)
        logits = tf.reshape(logits, [-1, self._num_actions, self._num_atoms])
        return logits, network_state
