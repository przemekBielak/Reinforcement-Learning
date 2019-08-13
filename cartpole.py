import gym 
import random
import numpy as np
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import Adam


# action: 0 - move lft, 1 - move right

env = gym.make('CartPole-v1')
env.reset()
number_of_games = 10000
steps_goal = 500
score_requirement = 60


def prepare_model_data():
    data_train = []
    accepted_scores = []

    for game_index in range(number_of_games):
        score = 0
        prev_obs = []
        game_memory = []

        for step_index in range(steps_goal):
            action = env.action_space.sample()
            obs, rew, done, info = env.step(action)
            if len(prev_obs) >  0:
                game_memory.append([prev_obs, action])

            prev_obs = obs
            score += rew
            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                data_train.append(data)

        env.reset()

    print(data_train)

    return data_train 

prepare_model_data()
