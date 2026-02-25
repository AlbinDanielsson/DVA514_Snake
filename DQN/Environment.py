import random
import sys
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

# game params
GRID_W, GRID_H = 10, 10
CELL_SIZE = 50
MARGIN = 2
FPS = 20

WINDOW_W = GRID_W * CELL_SIZE
WINDOW_H = GRID_H * CELL_SIZE

# Colors (R,G,B)
BG = (90, 200, 90)
GRID_LINE = (40, 40, 48)
SNAKE_HEAD = (150, 75, 0)
SNAKE_BODY = (10, 60, 10)
FOOD = (220, 10, 10)
TEXT = (230, 230, 240)


def draw_board(screen, font, table, moves):
    screen.fill(BG)

    for x in range(GRID_W):
        for y in range(GRID_H):
            r = pygame.Rect(
                x * CELL_SIZE + MARGIN,
                y * CELL_SIZE + MARGIN,
                CELL_SIZE - 2 * MARGIN,
                CELL_SIZE - 2 * MARGIN,
            )

            cell = table[x][y]

            if cell == 0:
                pygame.draw.rect(screen, BG, r)
            elif cell == -1:
                pygame.draw.rect(screen, FOOD, r, border_radius=8)
            elif cell == 1:
                pygame.draw.rect(screen, SNAKE_HEAD, r, border_radius=8)
            else:
                pygame.draw.rect(screen, SNAKE_BODY, r, border_radius=8)

    # Optional: draw grid lines on top
    for x in range(GRID_W):
        for y in range(GRID_H):
            grid_rect = pygame.Rect(
                x * CELL_SIZE,
                y * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            )
            pygame.draw.rect(screen, GRID_LINE, grid_rect, 1)

    hud = f"moves: {moves}"
    surf = font.render(hud, True, TEXT)
    screen.blit(surf, (8, 8))


def newHeadPos(dir, posX, posY):
    if dir == 0:  # up
        return (posX, posY - 1)
    elif dir == 1:  # right
        return (posX + 1, posY)
    elif dir == 2:  # down
        return (posX, posY + 1)
    else:  # dir == 3 left
        return (posX - 1, posY)


def spawnFood(snake, lenght):
    occupied = set(snake[:lenght])
    free_cells = [(x, y) for x in range(GRID_W) for y in range(GRID_H) if (x, y) not in occupied]
    if not free_cells:
        return None  # grid full => win
    return random.choice(free_cells)


class SnakeDqnEnv(gym.Env):
    metadata = {"render_fps": FPS}

    def __init__(self):
        super().__init__()

        # Actions: relative turn (CCW, straight, CW)
        self.action_space = spaces.Discrete(3)

        # Observation: table with values in [-1 .. GRID_W*GRID_H]
        self.observation_space = spaces.Box(
            low=-1,
            high=GRID_W * GRID_H,
            shape=(GRID_W * GRID_H + 1,), #Number of inputs
            dtype=np.int8
        )

        # Pygame rendering
        self.screen = None
        self.clock = None
        self.font = None

        # State
        self.snake = None
        self.trail = None
        self.length = None
        self.food = None
        self.moves = 0
        self.totalMoves = 0
        self.isDead = False
        self.steps = 0

        self.visible = True

        # Cached table for render/obs
        self.table = None

    def _build_table(self):
        table = [[0 for _ in range(GRID_H)] for _ in range(GRID_W)]
        if self.food is not None:
            table[self.food[0]][self.food[1]] = -1
        for i in range(self.length):
            x, y = self.snake[i]
            if 0 <= x < GRID_W and 0 <= y < GRID_H:
                table[x][y] = i + 1
        self.table = table

    def _get_obs(self):
        grid = np.asarray(self.table, dtype=np.int8).reshape(-1)   # (100,)
        moves = np.asarray([self.moves], dtype=np.int8)            # (1,)
        return np.concatenate([grid, moves])                       # (101,)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # The snake itself (same as your init)
        self.snake = [(2, 2), (1, 2), (0, 2)]  # head first
        self.length = 3

        # 0=up,1=right,2=down,3=left ; trail stores absolute direction
        self.trail = [1, 1]

        self.food = spawnFood(self.snake, self.length)

        self.totalMoves = 0
        self.moves = 0
        self.isDead = False
        self.steps = 0

        self._build_table()

        if self.visible:
            self.render()

        info = {}
        return self._get_obs(), info



    def step(self, action):
        self.steps += 1
        self.moves += 1

        # Map action {0,1,2} -> turn {-1,0,+1} (CCW, straight, CW)
        if action == 0:
            turn = -1
        elif action == 1:
            turn = 0
        else:
            turn = 1

        # Current heading is trail[0]
        direction = (turn + self.trail[0] + 4) % 4

        newHead = newHeadPos(direction, self.snake[0][0], self.snake[0][1])
        tail = self.snake[self.length - 1]
        tT = self.trail[self.length - 2]

        # Move snake
        for i in range(self.length - 1, 0, -1):
            self.snake[i] = self.snake[i - 1]
        self.snake[0] = newHead

        # Store how it moved
        for i in range(self.length - 2, 0, -1):
            self.trail[i] = self.trail[i - 1]
        self.trail[0] = direction

        reward = -0.01

        terminated = False
        truncated = False

        # Grow
        if self.food is not None and newHead == self.food:
            self.length += 1
            self.snake.append(tail)
            self.trail.append(tT)

            self.food = spawnFood(self.snake, self.length)
            self.totalMoves += self.moves
            self.moves = 0

            reward += 10

        # Out of bounds
        if (newHead[0] > GRID_W - 1) or (newHead[0] < 0):
            self.isDead = True
        if (newHead[1] > GRID_H - 1) or (newHead[1] < 0):
            self.isDead = True

        # Missed deadline
        deadline = GRID_H * GRID_W
        if self.moves >= deadline:
            self.isDead = True

        # Self collision
        for i in range(1, self.length):
            if self.snake[0] == self.snake[i]:
                self.isDead = True
                break

        if self.isDead:
            self.totalMoves += self.moves
            terminated = True
            reward -= 10

        self._build_table()

        if self.visible:
            self.render()

        info = {"moves": self.moves, "length": self.length, "totalMoves": self.totalMoves}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
            pygame.display.set_caption("SnakeEnv")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("consolas", 18)

        # Basic quit handling when rendering
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit(0)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close()
                sys.exit(0)

        draw_board(self.screen, self.font, self.table, self.moves)
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        return None

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None