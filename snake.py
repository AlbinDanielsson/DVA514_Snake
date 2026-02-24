import random
import sys
from collections import deque

import pygame

#game params
GRID_W, GRID_H = 10, 10
CELL_SIZE = 50
MARGIN = 2
FPS = 8

WINDOW_W = GRID_W * CELL_SIZE
WINDOW_H = GRID_H * CELL_SIZE

#Colors (R,G,B)
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

def spawnFood(snake, lenght):
    # Build a set of occupied cells (snake positions)
    occupied = set(snake[:lenght])

    # Fast path: choose from all free cells
    free_cells = [(x, y) for x in range(GRID_W) for y in range(GRID_H)
                  if (x, y) not in occupied]

    if not free_cells:
        return None  # grid is full; you can treat this as "win"

    return random.choice(free_cells)

def randomDirection():
    x = random.randrange(-1, 2) #-1, 0 or 1
    return x

def playGame(showGame):
    if showGame:
        screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("consolas", 18)

    #The snake itself
    snake = []
    snake.append((2, 2)) #Head
    snake.append((1, 2))
    snake.append((0, 2))
    snake.append((0, 1))
    snake.append((0, 0))
    length = 5

    #0 = up, 1 = right, 2 = down, 3 = left
    trail = [] #length of snake -1
    trail.append(1)
    trail.append(1)
    trail.append(1)
    trail.append(1)

    #Spawn food, TODO make random
    food = spawnFood(snake, length)

    totalMoves = 0
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
        #1CW, 0 forward, -1 CCW
        direction = randomDirection()
        print('got direction: ', direction)
        direction = (direction + trail[0] + 4) % 4
        newHead = newHeadPos(direction, snake[0][0], snake[0][1])
        tail = snake[length - 1]
        tT = trail[length - 2]

        #Move snake
        for i in range(length - 1, 0, -1):
            snake[i] = snake[i - 1]
        snake[0] = newHead

        #Store how it moved
        for i in range(length - 2, 0, -1):
            trail[i] = trail[i - 1]
        trail[0] = direction

        #Grow
        if newHead == food:
            length = length + 1
            snake.append(tail)
            trail.append(tT)
    
            #Respawn food
            food = spawnFood(snake, length)

            totalMoves += moves
            moves = 0

        #Matrix representation of the board
        table = [[0 for _ in range(GRID_H)] for _ in range(GRID_W)]
        table[food[0]][food[1]] = -1
        for i in range (length):
            x, y = snake[i]
            if 0 <= x < GRID_W and 0 <= y < GRID_H:
                table[x][y] = i + 1

        if showGame:
            draw_board(screen, font, table, moves)
            pygame.display.flip()
            clock.tick(FPS)

        #Out of bounds
        if (newHead[0] > GRID_W - 1) | (newHead[0] < 0):
            isDead = True
        if (newHead[1] > GRID_H - 1)| (newHead[1] < 0):
            isDead = True

        #Missed deadline
        if moves >= GRID_H * GRID_W:
            isDead = True
        
        #Self Collision
        for i in range(1, length, 1):
            if snake[0] == snake[i]:
                isDead = True
                
        #Check for death
        if isDead:
            totalMoves += moves
            break

    return moves, length
        
def main():
    pygame.init()

    showGame = True
    moves, length = playGame(showGame)

    print('moves =', moves, 'eaten apples = ', length - 3)

if __name__ == "__main__":
    main()
