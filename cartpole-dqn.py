import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

env = gym.make('CartPole-v0')

model = Sequential()
model.add(
    Dense(24, input_dim=env.observation_space.shape[0], activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
model.compile(loss='mse', optimizer=Adam(), metrics=['mae'])

model.load_weights("weights2.h5")


def train():
    gamma = 1.0
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.999
    all_rewards = []

    total_episodes = 1000
    batch_size = 64
    memory_size = 50000
    memory = []

    for episode in range(total_episodes):
        state = env.reset()
        state = np.array([state])
        episode_reward = 0

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        done = False
        while not done:
            # env.render()

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(model.predict(state))

            next_state, reward, done, _ = env.step(action)
            next_state = np.array([next_state])
            episode_reward += reward

            memory.append((state, action, reward, next_state, done))
            state = next_state

            # free first items in memory
            if len(memory) == memory_size:
                del memory[:5000]

        all_rewards.append(episode_reward)
        print(episode, '\t', str(episode_reward), epsilon)

        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            for state, action, reward, next_state, done in minibatch:
                if not done:
                    target = reward + gamma * np.max(model.predict(next_state))
                else:
                    target = reward
                target_f = model.predict(state)[0]
                target_f[action] = target
                model.fit(state, target_f.reshape(-1, env.action_space.n),
                          epochs=1, verbose=0)

    model.save_weights("weights2.h5")
    plt.plot(all_rewards)
    plt.show()


def play(mode):
    state = env.reset()
    done = False
    while not done:
        env.render()
        if mode == 'trained':
            action = np.argmax(model.predict(np.array([state])))
        elif mode == 'random':
            action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        state = next_state

    env.close()


def main():
    train()
    play('trained')


if __name__ == '__main__':
    main()
