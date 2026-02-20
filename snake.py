import random
import sys
from collections import deque

import pygame

#game params
GRID_W, GRID_H = 10, 10
CELL_SIZE = 50
MARGIN = 2
FPS = 10

WINDOW_W = GRID_W * CELL_SIZE
WINDOW_H = GRID_H * CELL_SIZE

#Colors (R,G,B)
BG = (90, 200, 90)
GRID_LINE = (40, 40, 48)
SNAKE_HEAD = (150, 75, 0)
SNAKE_BODY = (10, 60, 10)
FOOD = (220, 10, 10)
TEXT = (230, 230, 240)

def draw_board(screen, font, snake, food, moves):
    screen.fill(BG)
    # grid + cells
    for x in range(GRID_W):
        for y in range(GRID_H):
            r = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GRID_LINE, r, 1)

    # snake
    for i, (sx, sy) in enumerate(snake):
        r = pygame.Rect(
            sx * CELL_SIZE + MARGIN,
            sy * CELL_SIZE + MARGIN,
            CELL_SIZE - 2 * MARGIN,
            CELL_SIZE - 2 * MARGIN,
        )
        pygame.draw.rect(screen, SNAKE_HEAD if i == 0 else SNAKE_BODY, r, border_radius=8)

    #food
    fx = food[0]
    fy = food[1]
    r = pygame.Rect(
            fx * CELL_SIZE + MARGIN,
            fy * CELL_SIZE + MARGIN,
            CELL_SIZE - 2 * MARGIN,
            CELL_SIZE - 2 * MARGIN,)
    pygame.draw.rect(screen, FOOD, r, border_radius=8)


    # HUD text (top-left overlay)
    hud = f"moves: {moves}"
    surf = font.render(hud, True, TEXT)
    screen.blit(surf, (8, 8))

def newHeadPos(dir, posX, posY):
    if dir == 0:
        newPosX = posX
        newPosY = posY - 1
    elif dir == 1:
        newPosX = posX + 1
        newPosY = posY
    elif dir == 2:
        newPosX = posX
        newPosY = posY + 1
    elif dir == 3:
        newPosX = posX - 1
        newPosY = posY
    return(newPosX, newPosY)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    #The snake itself
    snake = []
    snake.append((2, 0)) #Head
    snake.append((1, 0))
    snake.append((0, 0))
    lenght = 3

    #0 = up, 1 = right, 2 = down, 3 = left
    trail = [] #length of snake -1
    trail.append(1)
    trail.append(1)

    #Spawn food, TODO make random
    food = (7, 0)

    moves = 0
    isDead = False

    while True:
        #For exiting, ecs key or red cross
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)

        moves = moves + 1

        #Get direction
        direction = 1#TODO, replace later
        direction = (direction + trail[0]) % 4

        #Move snake
        for i in range(lenght - 1, 0, -1):
            snake[i] = snake[i - 1]
        snake[0] = newHeadPos(direction, snake[0][0], snake[0][1])

        for i in range(lenght - 2, 0, -1):
            trail[i] = trail[i - 1]
        trail[0] = direction
        
        #Or grow if it landed on food
        #Respawn food
        
        draw_board(screen, font, list(snake), food, direction)
        pygame.display.flip()
        clock.tick(FPS)

        #Check for death
        if isDead:
            break

if __name__ == "__main__":
    main()
