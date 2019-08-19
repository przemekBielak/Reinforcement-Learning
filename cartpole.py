import gym
import numpy as np
import random

alpha = 0.2
gamma = 0.9
epsilon = 0
num_of_episodes = 10000


class RandomAgent:
    def __init__(self, env_name, max_eps):
        self.env = gym.make(env_name)
        self.max_episodes = max_eps

    def run(self):
        for episode in range(self.max_episodes):
            state = self.env.reset()

            done = False
            while not done:
                action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)
                self.env.render()

                if done:
                    break

        self.env.close()


def train():
    for _ in range(num_of_episodes):
        state = env.reset()
        total_reward = 0
        epsilon *= 0.9997

        done = False
        while not done:
            # env.render()

            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            next_state, reward, done, _ = env.step(action)

            Q[state, action] = (1 - alpha) * Q[state, action] + \
                alpha * (reward + gamma *
                         np.max(Q[next_state, :] - Q[state, :]))

            total_reward += reward
            state = next_state

    print('final Q value: {}'.format(Q))


def main():
    agent = RandomAgent('CartPole-v0', 1000)
    agent.run()


if __name__ == '__main__':
    main()
