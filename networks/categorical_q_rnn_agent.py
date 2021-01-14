from typing import Optional, Text

import gin
import tensorflow as tf

from tf_agents.agents import tf_agent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import network
from tf_agents.networks import utils as network_utils
from tf_agents.policies import boltzmann_policy
from tf_agents.policies import categorical_q_policy
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.policies import greedy_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import nest_utils
from tf_agents.utils import value_ops


@gin.configurable
class CategoricalQRnnAgent(dqn_agent.DqnAgent):
    def __init__(
        self,
        time_step_spec: ts.TimeStep,
        action_spec: types.NestedTensorSpec,
        categorical_q_network: network.Network,
        optimizer: types.Optimizer,
        observation_and_action_constraint_splitter: Optional[
            types.Splitter] = None,
        min_q_value: types.Float = -10.0,
        max_q_value: types.Float = 10.0,
        epsilon_greedy: types.Float = 0.1,
        n_step_update: int = 1,
        boltzmann_temperature: Optional[types.Float] = None,
        # Params for target network updates
        target_categorical_q_network: Optional[network.Network] = None,
        target_update_tau: types.Float = 1.0,
        target_update_period: types.Int = 1,
        # Params for training.
        td_errors_loss_fn: Optional[types.LossFn] = None,
        gamma: types.Float = 1.0,
        reward_scale_factor: types.Float = 1.0,
        gradient_clipping: Optional[types.Float] = None,
        # Params for debugging
        debug_summaries: bool = False,
        summarize_grads_and_vars: bool = False,
        train_step_counter: Optional[tf.Variable] = None,
            name: Optional[Text] = None):
        def check_atoms(net, label):
            try:
                num_atoms = net.num_atoms
            except AttributeError:
                raise TypeError('Expected {} to have property `num_atoms`, but it '
                                'doesn\'t. (Note: you likely want to use a '
                                'CategoricalQNetwork.) Network is: {}'.format(
                                    label, net))
            return num_atoms

        self._num_atoms = check_atoms(
            categorical_q_network, 'categorical_q_network')

        if target_categorical_q_network is not None:
            target_num_atoms = check_atoms(
                target_categorical_q_network, 'target_categorical_q_network')
            if self._num_atoms != target_num_atoms:
                raise ValueError(
                    'categorical_q_network and target_categorical_q_network have '
                    'different numbers of atoms: {} vs. {}'.format(
                        self._num_atoms, target_num_atoms))

        self._min_q_value = min_q_value
        self._max_q_value = max_q_value
        min_q_value = tf.convert_to_tensor(min_q_value, dtype_hint=tf.float32)
        max_q_value = tf.convert_to_tensor(max_q_value, dtype_hint=tf.float32)
        self._support = tf.linspace(min_q_value, max_q_value, self._num_atoms)

        super(CategoricalQRnnAgent, self).__init__(
            time_step_spec,
            action_spec,
            categorical_q_network,
            optimizer,
            observation_and_action_constraint_splitter=(
                observation_and_action_constraint_splitter),
            epsilon_greedy=epsilon_greedy,
            n_step_update=n_step_update,
            boltzmann_temperature=boltzmann_temperature,
            target_q_network=target_categorical_q_network,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=td_errors_loss_fn,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter,
            name=name)

    def _setup_policy(self, time_step_spec, action_spec,
                      boltzmann_temperature, emit_log_probability):
        policy = categorical_q_policy.CategoricalQPolicy(
            time_step_spec,
            action_spec,
            self._q_network,
            self._min_q_value,
            self._max_q_value,
            observation_and_action_constraint_splitter=(
                self._observation_and_action_constraint_splitter))

        if boltzmann_temperature is not None:
            collect_policy = boltzmann_policy.BoltzmannPolicy(
                policy, temperature=boltzmann_temperature)
        else:
            collect_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(
                policy, epsilon=self._epsilon_greedy)
        policy = greedy_policy.GreedyPolicy(policy)

        target_policy = categorical_q_policy.CategoricalQPolicy(
            time_step_spec,
            action_spec,
            self._target_q_network,
            self._min_q_value,
            self._max_q_value,
            observation_and_action_constraint_splitter=(
                self._observation_and_action_constraint_splitter))
        self._target_greedy_policy = greedy_policy.GreedyPolicy(target_policy)

        return policy, collect_policy

    def _check_network_output(self, net, label):
        network_utils.check_single_floating_network_output(
            net.create_variables(),
            expected_output_shape=(self._num_actions, self._num_atoms),
            label=label)

    def _loss(self,
              experience,
              td_errors_loss_fn=tf.compat.v1.losses.huber_loss,
              gamma=1.0,
              reward_scale_factor=1.0,
              weights=None,
              training=False):
        self._check_trajectory_dimensions(experience)

        squeeze_time_dim = not self._q_network.state_spec
        if self._n_step_update == 1:
            time_steps, policy_steps, next_time_steps = (
                trajectory.experience_to_transitions(experience, squeeze_time_dim))
            actions = policy_steps.action
        else:
            first_two_steps = tf.nest.map_structure(
                lambda x: x[:, :2], experience)
            last_two_steps = tf.nest.map_structure(
                lambda x: x[:, -2:], experience)
            time_steps, policy_steps, _ = (
                trajectory.experience_to_transitions(
                    first_two_steps, squeeze_time_dim))
            actions = policy_steps.action
            _, _, next_time_steps = (
                trajectory.experience_to_transitions(
                    last_two_steps, squeeze_time_dim))

        with tf.name_scope('critic_loss'):
            nest_utils.assert_same_structure(actions, self.action_spec)
            nest_utils.assert_same_structure(time_steps, self.time_step_spec)
            nest_utils.assert_same_structure(
                next_time_steps, self.time_step_spec)

            rank = nest_utils.get_outer_rank(time_steps.observation,
                                             self._time_step_spec.observation)

            # batch_squash = (None
            #                 if rank <= 1 or self._q_network.state_spec in ((), None)
            #                 else network_utils.BatchSquash(rank))
            batch_squash = None

            network_observation = time_steps.observation

            if self._observation_and_action_constraint_splitter is not None:
                network_observation, _ = (
                    self._observation_and_action_constraint_splitter(
                        network_observation))

            q_logits, _ = self._q_network(network_observation,
                                          step_type=time_steps.step_type,
                                          training=training)
            print(q_logits)
            print(actions)
            if batch_squash is not None:
                q_logits = batch_squash.flatten(q_logits)
                actions = batch_squash.flatten(actions)
                next_time_steps = tf.nest.map_structure(batch_squash.flatten,
                                                        next_time_steps)

            next_q_distribution = self._next_q_distribution(next_time_steps)

            if actions.shape.rank > 1:
                actions = tf.squeeze(actions, list(
                    range(1, actions.shape.rank)))

            batch_size = q_logits.shape[0] or tf.shape(q_logits)[0]
            tiled_support = tf.tile(self._support, [batch_size])
            tiled_support = tf.reshape(
                tiled_support, [batch_size, self._num_atoms])

            if self._n_step_update == 1:
                discount = next_time_steps.discount
                if discount.shape.rank == 1:
                    discount = tf.expand_dims(discount, -1)
                print(discount)
                print(tiled_support)
                next_value_term = tf.multiply(discount,
                                              tiled_support,
                                              name='next_value_term')
                print(next_value_term)
                exit(0)

                reward = next_time_steps.reward
                if reward.shape.rank == 1:
                    reward = tf.expand_dims(reward, -1)
                reward_term = tf.multiply(reward_scale_factor,
                                          reward,
                                          name='reward_term')

                target_support = tf.add(reward_term, gamma * next_value_term,
                                        name='target_support')
            else:
                rewards = reward_scale_factor * experience.reward[:, :-1]
                discounts = gamma * experience.discount[:, :-1]

                discounted_returns = value_ops.discounted_return(
                    rewards=rewards,
                    discounts=discounts,
                    final_value=tf.zeros([batch_size], dtype=discounts.dtype),
                    time_major=False,
                    provide_all_returns=False)

                discounted_returns = tf.expand_dims(discounted_returns, -1)

                final_value_discount = tf.reduce_prod(discounts, axis=1)
                final_value_discount = tf.expand_dims(final_value_discount, -1)

                self._discounted_returns = discounted_returns
                self._final_value_discount = final_value_discount

                target_support = tf.add(discounted_returns,
                                        final_value_discount * tiled_support,
                                        name='target_support')

            target_distribution = tf.stop_gradient(project_distribution(
                target_support, next_q_distribution, self._support))

            indices = tf.range(batch_size)
            indices = tf.cast(indices, actions.dtype)
            reshaped_actions = tf.stack([indices, actions], axis=-1)
            chosen_action_logits = tf.gather_nd(q_logits, reshaped_actions)

            if batch_squash is not None:
                target_distribution = batch_squash.unflatten(
                    target_distribution)
                chosen_action_logits = batch_squash.unflatten(
                    chosen_action_logits)
                critic_loss = tf.reduce_sum(
                    tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
                        labels=target_distribution,
                        logits=chosen_action_logits),
                    axis=1)
            else:
                critic_loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
                    labels=target_distribution,
                    logits=chosen_action_logits)

            agg_loss = common.aggregate_losses(
                per_example_loss=critic_loss,
                regularization_loss=self._q_network.losses)
            total_loss = agg_loss.total_loss

            dict_losses = {'critic_loss': agg_loss.weighted,
                           'reg_loss': agg_loss.regularization,
                           'total_loss': total_loss}

            common.summarize_scalar_dict(dict_losses,
                                         step=self.train_step_counter,
                                         name_scope='Losses/')

            if self._debug_summaries:
                distribution_errors = target_distribution - chosen_action_logits
                with tf.name_scope('distribution_errors'):
                    common.generate_tensor_summaries(
                        'distribution_errors', distribution_errors,
                        step=self.train_step_counter)
                    tf.compat.v2.summary.scalar(
                        'mean', tf.reduce_mean(distribution_errors),
                        step=self.train_step_counter)
                    tf.compat.v2.summary.scalar(
                        'mean_abs', tf.reduce_mean(
                            tf.abs(distribution_errors)),
                        step=self.train_step_counter)
                    tf.compat.v2.summary.scalar(
                        'max', tf.reduce_max(distribution_errors),
                        step=self.train_step_counter)
                    tf.compat.v2.summary.scalar(
                        'min', tf.reduce_min(distribution_errors),
                        step=self.train_step_counter)
                with tf.name_scope('target_distribution'):
                    common.generate_tensor_summaries(
                        'target_distribution', target_distribution,
                        step=self.train_step_counter)

            return tf_agent.LossInfo(total_loss, dqn_agent.DqnLossInfo(td_loss=(),
                                                                       td_error=()))

    def _next_q_distribution(self, next_time_steps):
        network_observation = next_time_steps.observation

        if self._observation_and_action_constraint_splitter is not None:
            network_observation, _ = self._observation_and_action_constraint_splitter(
                network_observation)

        next_target_logits, _ = self._target_q_network(
            network_observation,
            step_type=next_time_steps.step_type,
            training=False)
        batch_size = next_target_logits.shape[0] or tf.shape(next_target_logits)[
            0]
        next_target_probabilities = tf.nn.softmax(next_target_logits)
        next_target_q_values = tf.reduce_sum(
            self._support * next_target_probabilities, axis=-1)
        dummy_state = self._target_greedy_policy.get_initial_state(batch_size)
        greedy_actions = self._target_greedy_policy.action(
            next_time_steps, dummy_state).action
        next_qt_argmax = tf.cast(greedy_actions, tf.int32)[:, None]
        batch_indices = tf.range(
            tf.cast(tf.shape(next_target_q_values)[0], tf.int32))[:, None]
        next_qt_argmax = tf.concat([batch_indices, next_qt_argmax], axis=-1)
        return tf.gather_nd(next_target_probabilities, next_qt_argmax)


def project_distribution(supports: types.Tensor,
                         weights: types.Tensor,
                         target_support: types.Tensor,
                         validate_args: bool = False) -> types.Tensor:
    target_support_deltas = target_support[1:] - target_support[:-1]
    delta_z = target_support_deltas[0]
    validate_deps = []
    supports.shape.assert_is_compatible_with(weights.shape)
    supports[0].shape.assert_is_compatible_with(target_support.shape)
    target_support.shape.assert_has_rank(1)
    if validate_args:
        validate_deps.append(
            tf.Assert(
                tf.reduce_all(tf.equal(tf.shape(supports), tf.shape(weights))),
                [supports, weights]))
        validate_deps.append(
            tf.Assert(
                tf.reduce_all(
                    tf.equal(tf.shape(supports)[1], tf.shape(target_support))),
                [supports, target_support]))
        validate_deps.append(
            tf.Assert(
                tf.equal(tf.size(tf.shape(target_support)), 1), [target_support]))
        validate_deps.append(
            tf.Assert(tf.reduce_all(target_support_deltas > 0), [target_support]))
        validate_deps.append(
            tf.Assert(
                tf.reduce_all(tf.equal(target_support_deltas, delta_z)),
                [target_support]))

    with tf.control_dependencies(validate_deps):
        v_min, v_max = target_support[0], target_support[-1]
        batch_size = tf.shape(supports)[0]
        num_dims = tf.shape(target_support)[0]
        clipped_support = tf.clip_by_value(supports, v_min, v_max)[:, None, :]
        tiled_support = tf.tile([clipped_support], [1, 1, num_dims, 1])
        reshaped_target_support = tf.tile(
            target_support[:, None], [batch_size, 1])
        reshaped_target_support = tf.reshape(reshaped_target_support,
                                             [batch_size, num_dims, 1])
        numerator = tf.abs(tiled_support - reshaped_target_support)
        quotient = 1 - (numerator / delta_z)
        clipped_quotient = tf.clip_by_value(quotient, 0, 1)
        weights = weights[:, None, :]
        inner_prod = clipped_quotient * weights
        projection = tf.reduce_sum(inner_prod, 3)
        projection = tf.reshape(projection, [batch_size, num_dims])
        return projection
