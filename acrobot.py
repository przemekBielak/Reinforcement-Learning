from gym import envs
import gym


env = gym.make('Acrobot-v1')
env.reset()

for _ in range(500):
    env.render()
    env.step(env.action_space.sample())

print(envs.registry.all())
