import numpy as np
from env import Pendulum

env = Pendulum.PyEnv()
while True:
    action = [np.random.choice(env._action_ref)]
    env.step(action)
    observation = env.get_state()
    print(observation)
    env.render()
