import tensorflow as tf
import cv2 as cv
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory


class ReplayBuffer:
    def __init__(self, data_spec, batch_size=1):
        self.data_spec = data_spec
        self.batch_size = batch_size
        self.replay_buffer_capacity = 100
        self.buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.data_spec,
            batch_size=self.batch_size,
            max_length=self.replay_buffer_capacity)
        # For Experience Filtering (EF)
        self.sub_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.data_spec,
            batch_size=self.batch_size,
            max_length=1000)

    def __len__(self):
        return self.buffer.num_frames()

    def clear(self):
        return self.buffer.clear()

    def as_dataset(self, sample_batch_size=64):
        return self.buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=sample_batch_size,
            num_steps=2
        ).prefetch(3)

    def _filtering(self, traj):
        # cv.imshow('OhmniInSpace-v0', traj.observation.numpy()[0])
        # cv.waitKey(10)
        if self.sub_buffer.num_frames() < 2:
            return False
        batch = tf.squeeze(self.sub_buffer.gather_all().observation)
        batch_observation = tf.tile(
            traj.observation, [len(batch), 1, 1, 1])
        min_distance = tf.reduce_min(tf.reduce_sum(
            tf.square(batch_observation-batch), axis=[-3, -2, -1]))
        threshold = 100
        if min_distance > threshold:
            return False
        else:
            return True

    def collect(self, env, policy):
        while True:
            time_step = env.current_time_step()
            action_step = policy.action(time_step)
            next_time_step = env.step(action_step.action)
            traj = trajectory.from_transition(
                time_step, action_step, next_time_step)
            if not self._filtering(traj):
                self.buffer.add_batch(traj)
                self.sub_buffer.add_batch(traj)
                return traj

    def collect_steps(self, env, policy, steps=1):
        for _ in range(steps):
            traj = self.collect(env, policy)
            if traj.is_boundary():
                self.sub_buffer.clear()
