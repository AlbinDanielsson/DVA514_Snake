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
FPS = 15

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

        # Dict observation:
        # - image: (C,H,W) uint8 masks
        #   channel 0: food, channel 1: head, channel 2: body
        # - state: small scalars (direction, moves_clipped)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(low=0, high=255, shape=(3, GRID_H, GRID_W), dtype=np.uint8),
                "state": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            }
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

        # Cached table for render
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
        img = np.zeros((3, GRID_H, GRID_W), dtype=np.uint8)
    
        # food
        if self.food is not None:
            fx, fy = self.food
            if 0 <= fx < GRID_W and 0 <= fy < GRID_H:
                img[0, fy, fx] = 255
    
        # head
        hx, hy = self.snake[0]
        if 0 <= hx < GRID_W and 0 <= hy < GRID_H:
            img[1, hy, hx] = 255
    
        # body
        for i in range(1, self.length):
            x, y = self.snake[i]
            img[2, y, x] = 255#20 + (self.length - i) * 2
            if i == self.length - 1:
                img[2, y, x] = 125


    
        direction = float(self.trail[0]) / 3.0
        moves_clip = float(min(self.moves, GRID_W * GRID_H)) / float(GRID_W * GRID_H)
        state = np.array([direction, moves_clip], dtype=np.float32)
    
        return {"image": img, "state": state}

    def willCrash(self, turn):
        deadly = False
        direction = (turn + self.trail[0] + 4) % 4
        newHead = newHeadPos(direction, self.snake[0][0], self.snake[0][1])

        # Out of bounds
        if (newHead[0] > GRID_W - 1) or (newHead[0] < 0):
            deadly = True
        if (newHead[1] > GRID_H - 1) or (newHead[1] < 0):
            deadly = True

        # Self collision
        for i in range(1, self.length - 1):
            if newHead == self.snake[i]:
                deadly = True
                break

        #Running out of time:
        if abs(newHead[0] - self.food[0]) + abs(newHead[1] - self.food[1]) > 100 - self.moves:
            deadly = True

        return deadly

    def reset(self, seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.length = random.randrange(3, 90)
        self.length = 3
        hamilton = [
        (1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0),(8,0),(9,0),
        (9,1),(8,1),(7,1),(6,1),(5,1),(4,1),(3,1),(2,1),(1,1),
        (1,2),(2,2),(3,2),(4,2),(5,2),(6,2),(7,2),(8,2),(9,2),
        (9,3),(8,3),(7,3),(6,3),(5,3),(4,3),(3,3),(2,3),(1,3),
        (1,4),(2,4),(3,4),(4,4),(5,4),(6,4),(7,4),(8,4),(9,4),
        (9,5),(8,5),(7,5),(6,5),(5,5),(4,5),(3,5),(2,5),(1,5),
        (1,6),(2,6),(3,6),(4,6),(5,6),(6,6),(7,6),(8,6),(9,6),
        (9,7),(8,7),(7,7),(6,7),(5,7),(4,7),(3,7),(2,7),(1,7),
        (1,8),(2,8),(3,8),(4,8),(5,8),(6,8),(7,8),(8,8),(9,8),
        (9,9),(8,9),(7,9),(6,9),(5,9),(4,9),(3,9),(2,9),(1,9),
        (0,9),(0,8),(0,7),(0,6),(0,5),(0,4),(0,3),(0,2),(0,1),(0,0)
        ]
        self.snake = hamilton[:self.length]
        self.snake.reverse()

        hamiltonDirections = [
            1, 1, 1, 1, 1, 1, 1, 1, 2,
            3, 3, 3, 3, 3, 3, 3, 3, 2,
            1, 1, 1, 1, 1, 1, 1, 1, 2,
            3, 3, 3, 3, 3, 3, 3, 3, 2,
            1, 1, 1, 1, 1, 1, 1, 1, 2,
            3, 3, 3, 3, 3, 3, 3, 3, 2,
            1, 1, 1, 1, 1, 1, 1, 1, 2,
            3, 3, 3, 3, 3, 3, 3, 3, 2,
            1, 1, 1, 1, 1, 1, 1, 1, 2,
            3, 3, 3, 3, 3, 3, 3, 3, 3,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1
        ]
        self.trail = hamiltonDirections[:self.length-1]
        self.trail.reverse()

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

        #See if we die moving in the chosen direction, and change if needed
        if self.willCrash(turn):
            turn = 0
            #Yes, try 0
            if self.willCrash(turn):
                turn = -1
                #Still hit, try -1
                if self.willCrash(turn):
                    #Still hit try + 1
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

            reward += 1

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
            if newHead == self.snake[i]:
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