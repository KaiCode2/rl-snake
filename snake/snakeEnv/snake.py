import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pygame
import random


class SnakeEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    ACTION = ["N", "S", "E", "W"]

    def __init__(self, grid_size=(100, 100), mode=None):

        self.viewer = None

        self.snake_view = SnakeView2D(grid_size=grid_size, screen_size=(640, 640))

        self.grid_size = self.snake_view.grid_size

        # forward or backward in each dimension
        self.action_space = spaces.Discrete(2*len(self.grid_size))

        # observation is the x, y coordinate of the grid
        low = np.zeros(len(self.grid_size), dtype=int)
        high =  np.array(self.grid_size, dtype=int) - np.ones(len(self.grid_size), dtype=int)
        self.observation_space = spaces.Box(low, high)

        # initial condition
        self.state = None
        self.steps_beyond_done = None

        # Simulation related variables.
        self._seed()
        self._reset()

        # Just need to initialize the relevant attributes
        self._configure()

    # def __del__(self):
    #     self.snake_view.quit_game()

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        if isinstance(action, int):
            self.snake_view.move_snake(self.ACTION[action])
        else:
            self.snake_view.move_snake(action)



        done = self.snake_view.game_over

        if np.array_equal(self.snake_view.snake.snake_head, self.snake_view.goal):
            reward = 1
        else:
            reward = -0.1/(self.grid_size[0]*self.grid_size[1])

        self.state = self.snake_view.snake.snake_head

        info = {}

        return self.state, reward, done, info

    def _reset(self):
        self.snake_view.reset_snake()
        self.state = np.zeros(2)
        self.steps_beyond_done = None
        self.done = False
        return self.state

    def is_game_over(self):
        return self.snake_view.game_over

    def _render(self, mode="human", close=False):
        if close:
            self.snake_view.quit_game()

        return self.snake_view.update(mode)


class SnakeView2D:

    def __init__(self, grid_size=(100, 100), screen_size=(600, 600)):

        # PyGame configurations
        pygame.init()
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.__game_over = False

        self.__snake = SnakeState()
        self.grid_size = self.__snake.grid_size
        # to show the right and bottom border
        self.screen = pygame.display.set_mode(screen_size)
        self.__screen_size = tuple(map(sum, zip(screen_size, (-1, -1))))

        # Set the starting point
        self.__entrance = np.zeros(2, dtype=int)

        # Set the Goal
        self.__goal = np.array(self.grid_size) - np.array((1, 1)) # write goal

        # Create a background
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.background.fill((255, 255, 255))

        # Create a layer for the Grid
        self.grid_layer = pygame.Surface(self.screen.get_size()).convert_alpha()
        self.grid_layer.fill((0, 0, 0, 0,))

        # show the board
        self.__draw_board()

        # show the snake
        self.__draw_snake()

        # show the target
        self.__draw_target()

    def update(self, mode="human"):
        try:
            img_output = self.__view_update(mode)
            self.__controller_update()
        except Exception as e:
            self.__game_over = True
            self.quit_game()
            raise e
        else:
            return img_output

    def quit_game(self):
        try:
            self.__game_over = True
            pygame.display.quit()
            pygame.quit()
        except Exception:
            pass

    def move_snake(self, dir):
        if dir not in self.__snake.COMPASS.keys():
            raise ValueError("dir cannot be %s. The only valid dirs are %s."
                             % (str(dir), str(self.__snake.COMPASS.keys())))

        next_point = np.array([self.__snake.snake_head[0] + self.__snake.COMPASS[dir][0], self.__snake.snake_head[1] + self.__snake.COMPASS[dir][1]])
        if self.__snake.is_within_bound(next_point[0], next_point[1]) and not self.__snake.hit_self(next_point[0], next_point[1]):

            # update the drawing
            self.__draw_snake(transparency=0)

            self.__snake.snake_boxes.append(next_point)

            # move the snake
            self.__snake.snake_head = next_point
            if self.__snake == self.__snake.target:
                self.__snake.snake_length += 1
                self.__snake.new_target()
            else:
                self.__snake.snake_boxes = self.__snake.snake_boxes[1:]

            self.__draw_snake(transparency=255)
        else:
            self.__game_over = True

    def reset_snake(self):
        self.__draw_board()
        self.__draw_snake(transparency=0)
        self.__game_over = False
        self.__snake.snake_head = np.zeros(2, dtype=int)
        self.__snake.snake_length = 1
        self.__draw_snake(transparency=255)

    def __controller_update(self):
        if not self.__game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.__game_over = True
                    self.quit_game()

    def __view_update(self, mode="human"):
        if not self.__game_over:
            # update the robot's position
            self.__draw_target()
            self.__draw_snake()


            # update the screen
            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.grid_layer,(0, 0))

            if mode == "human":
                pygame.display.flip()

            return np.flipud(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface())))

    def __draw_board(self):

        line_colour = (0, 0, 0, 255)

        # drawing the horizontal lines
        for y in range(self.snake.GRID_H + 1):
            pygame.draw.line(self.grid_layer, line_colour, (0, y * self.CELL_H),
                             (self.SCREEN_W, y * self.CELL_H))

        # drawing the vertical lines
        for x in range(self.snake.GRID_W + 1):
            pygame.draw.line(self.grid_layer, line_colour, (x * self.CELL_W, 0),
                             (x * self.CELL_W, self.SCREEN_H))

    def __draw_snake(self, colour=(0, 0, 150), transparency=255):
        for box in self.__snake.snake_boxes:
            pygame.draw.rect(self.grid_layer, colour + (transparency,), [box[0] * self.CELL_W, box[1] * self.CELL_H, self.CELL_W, self.CELL_H])

    def __draw_target(self, colour=(150, 0, 0), transparency=235):

        self.__colour_cell(self.__snake.target, colour=colour, transparency=transparency)


    def __colour_cell(self, cell, colour, transparency):

        if not (isinstance(cell, (list, tuple, np.ndarray)) and len(cell) == 2):
            raise TypeError("cell must a be a tuple, list, or numpy array of size 2")

        x = int(cell[0] * self.CELL_W + 0.5 + 1)
        y = int(cell[1] * self.CELL_H + 0.5 + 1)
        w = int(self.CELL_W + 0.5 - 1)
        h = int(self.CELL_H + 0.5 - 1)
        pygame.draw.rect(self.grid_layer, colour + (transparency,), (x, y, w, h))

    @property
    def snake(self):
        return self.__snake

    @property
    def entrance(self):
        return self.__entrance

    @property
    def goal(self):
        return self.__goal

    @property
    def game_over(self):
        return self.__game_over

    @property
    def SCREEN_SIZE(self):
        return tuple(self.__screen_size)

    @property
    def SCREEN_W(self):
        return int(self.SCREEN_SIZE[0])

    @property
    def SCREEN_H(self):
        return int(self.SCREEN_SIZE[1])

    @property
    def CELL_W(self):
        return float(self.SCREEN_W) / float(self.snake.GRID_W)

    @property
    def CELL_H(self):
        return float(self.SCREEN_H) / float(self.snake.GRID_H)


class SnakeState:

    COMPASS = {
        "N": (0, -1),
        "E": (1, 0),
        "S": (0, 1),
        "W": (-1, 0)
    }

    def __init__(self, grid_size=(100, 100)):
        self.grid_size = grid_size
        self.snake_length = 1
        self.snake_head = np.zeros(2, dtype=int)
        self.target = (random.randint(0, self.GRID_W-1), random.randint(0, self.GRID_H-1))
        self.snake_boxes = [self.snake_head]

    def new_target(self):
        self.target = (random.randint(0, self.GRID_W-1), random.randint(0, self.GRID_H-1))

    def is_within_bound(self, x, y):
        # true if cell is still within bounds after the move
        return 0 <= x < self.GRID_W and 0 <= y < self.GRID_H

    def hit_self(self, x, y):
        for box in self.snake_boxes:
            if box[0] == x and box[1] == y:
                return True
        return False


    @property
    def GRID_W(self):
        return int(self.grid_size[0])

    @property
    def GRID_H(self):
        return int(self.grid_size[1])


if __name__ == "__main__":

    snake = SnakeView2D(screen_size= (500, 500), grid_size=(100, 100))
    snake.update()
    input("Enter any key to quit.")
