import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class Agent:
    def __init__(self, env_name, max_eps, gamma, learning_rate, epsilon):
        self.env = gym.make(env_name)
        self.max_episodes = max_eps
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        self.memory = deque(maxlen=100000)
        self.avarage_reward = 0
        self.episode_reward = 0
        self.all_rewards = []

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(self.env.observation_space.shape[0]), activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(self.env.action_space, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state)[0])

    def run(self):
        for _ in range(self.max_episodes):
            state = self.env.reset()

            done = False
            while not done:
                action = act(state)
                next_state, reward, done, _ = self.env.step(action)
                # self.env.render()
                self.avarage_reward += reward
                self.episode_reward += reward
                if done:
                    break

            self.all_rewards.append(self.episode_reward)
            self.episode_reward = 0
        self.avarage_reward /= self.max_episodes

        # plt.plot(self.all_rewards)
        # plt.show()

        self.env.close()
        return self.avarage_reward

    def learn(self):



def main():
    agent = Agent('CartPole-v0', 100, 0.95, 0.001, 1)
    print(agent.run())


if __name__ == '__main__':
    main()
