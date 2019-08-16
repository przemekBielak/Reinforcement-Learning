import gym
import numpy as np
import random


env = gym.make('FrozenLake8x8-v0')
Q = np.zeros([env.observation_space.n, env.action_space.n])
reward_list = []

alpha = 0.2
gamma = 0.9
epsilon = 0.1
num_of_episodes = 100

for episode in range(num_of_episodes):
    state = env.reset()
    total_reward = 0

    done = False
    while not done:
        # env.render()
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        next_state, reward, done, _ = env.step(action)

        Q[state, action] = (1 - alpha) * Q[state, action] + \
            alpha * (reward + gamma * np.max(Q[next_state, :] - Q[state, :]))

        total_reward += reward
        state = next_state

    reward_list.append(total_reward)

print('reward sum from all episodes: {}'.format(str(sum(reward_list))))
print('final Q value: {}'.format(Q))
