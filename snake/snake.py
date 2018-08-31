import gym
import snakeEnv

env = gym.make('Snake-v0')
for i_episode in range(200):
    observation = env.reset()
    for t in range(1000):
        env.render(mode="human")
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

class SnakeAgent:

    def __init__(self):
        self.environment = env = gym.make('Snake-v0')
