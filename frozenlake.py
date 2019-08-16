import gym
import numpy as np


env = gym.make('FrozenLake8x8-v0')
Q = np.zeros([env.observation_space.n, env.action_space.n])
reward_list = []

alpha = 0.628
gamma = 0.9
num_of_episodes = 5000

for episode in range(num_of_episodes):
    observation = env.reset()
    total_reward = 0

    for step in range(99):
        env.render()
        action = np.argmax(
            Q[observation, :] + np.random.randn(1, env.action_space.n) * (1.0 / (step + 1)))

        next_observation, reward, done, _ = env.step(action)
        Q[observation, action] = Q[observation, action] + alpha * \
            (reward + gamma * np.max(Q[next_observation, :]) - Q[observation, action])

        total_reward += reward
        observation = next_observation

        if done:
            break

        env.render()
    reward_list.append(total_reward)

print('reward sum from all episodes: {}'.format(str(sum(reward_list))))
print('final Q value: {}'.format(Q))
