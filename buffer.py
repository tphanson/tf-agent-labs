from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory


class ReplayBuffer:
    def __init__(self, data_spec, batch_size=1):
        self.data_spec = data_spec
        self.batch_size = batch_size
        self.replay_buffer_capacity = 100000
        self.buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.data_spec,
            batch_size=self.batch_size,
            max_length=self.replay_buffer_capacity,
        )
    def __len__(self):
        return self.buffer.num_frames()

    def clear(self):
        return self.buffer.clear()

    def as_dataset(self, sample_batch_size=32):
        return self.buffer.as_dataset(
            sample_batch_size=sample_batch_size,
            num_steps=3
        ).prefetch(3)

    def collect(self, env, policy):
        time_step = env.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = env.step(action_step.action)
        traj = trajectory.from_transition(
            time_step, action_step, next_time_step)
        self.buffer.add_batch(traj)
        return traj

    def collect_steps(self, env, policy, steps=1):
        for _ in range(steps):
            self.collect(env, policy)
