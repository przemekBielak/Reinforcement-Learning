import gym
import numpy as np


env = gym.make('FrozenLake8x8-v0')
observation = env.reset()

for t in range(50):
    env.render()
    print(observation)

    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(obs, reward, done, info)
    if done:
        print('finished after {} timesteps'.format(t + 1))
        break
