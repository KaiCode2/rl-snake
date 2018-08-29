from gym.envs.registration import register

register(
    id='Snake-v0',
    entry_point='snakeEnv.snake:SnakeEnv',
)
