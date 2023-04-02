import pygame
import random

# 初始化 Pygame
pygame.init()

# 游戏窗口大小和标题
WINDOW_WIDTH = 300
WINDOW_HEIGHT = 600
WINDOW_TITLE = "Tetris"

# 方块大小
BLOCK_SIZE = 30

# 游戏板大小
BOARD_WIDTH = 10
BOARD_HEIGHT = 20

# 游戏板位置
BOARD_POS_X = (WINDOW_WIDTH - BOARD_WIDTH * BLOCK_SIZE) // 2
BOARD_POS_Y = WINDOW_HEIGHT - BOARD_HEIGHT * BLOCK_SIZE

# 游戏板颜色
BOARD_COLOR = (0, 0, 0)

# 方块颜色
BLOCK_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 128, 128),
]

# 方向键控制
LEFT = pygame.K_LEFT
RIGHT = pygame.K_RIGHT
DOWN = pygame.K_DOWN
UP = pygame.K_UP

# 旋转键控制
ROTATE = pygame.K_SPACE

# 初始化游戏窗口
game_window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption(WINDOW_TITLE)

# 游戏时钟
clock = pygame.time.Clock()

# 定义方块类


class Block:
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = BLOCK_COLORS[shape]

    def draw(self):
        pygame.draw.rect(
            game_window,
            self.color,
            (
                BOARD_POS_X + self.x * BLOCK_SIZE,
                BOARD_POS_Y + self.y * BLOCK_SIZE,
                BLOCK_SIZE,
                BLOCK_SIZE,
            ),
        )

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def rotate(self):
        self.shape = (self.shape + 1) % 4


# 定义游戏板类


class Board:
    def __init__(self):
        self.board = [[None for _ in range(BOARD_WIDTH)]
                      for _ in range(BOARD_HEIGHT)]

    def draw(self):
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if self.board[y][x]:
                    pygame.draw.rect(
                        game_window,
                        self.board[y][x],
                        (
                            BOARD_POS_X + x * BLOCK_SIZE,
                            BOARD_POS_Y + y * BLOCK_SIZE,
                            BLOCK_SIZE,
                            BLOCK_SIZE,
                        ),
                    )

    def is_valid_position(self, block):
        for y in range(4):
            for x in range(4):
                if block.shape[y][x] != 0:
                    if (
                        block.x + x < 0
                        or block.x + x >= BOARD_WIDTH
                        or block.y + y >= BOARD_HEIGHT
                        or self.board[block.y + y][block.x + x]
                    ):
                        return False
        return True

    def add_block(self, block):
        for y in range(4):
            for x in range(4):
                if block.shape[y][x] != 0:
                    self.board[block.y + y][block.x + x] = block.color

    def remove_filled_rows(self):
        num_rows_removed = 0
        y = BOARD_HEIGHT - 1
        while y >= 0:
            if None not in self.board[y]:
                num_rows_removed += 1
                for yy in range(y, 0, -1):
                    for x in range(BOARD_WIDTH):
                        self.board[yy][x] = self.board[yy - 1][x]
                for x in range(BOARD_WIDTH):
                    self.board[0][x] = None
            else:
                y -= 1
            return num_rows_removed


# 生成新的方块


def generate_new_block():
    shapes = [
        [[1, 1, 1, 1]],
        [[1, 1, 0], [0, 1, 1]],
        [[0, 1, 1], [1, 1, 0]],
        [[1, 1], [1, 1]],
        [[1, 0, 0], [1, 1, 1]],
        [[0, 0, 1], [1, 1, 1]],
        [[1, 1, 1], [0, 1, 0]],
    ]
    shape = random.choice(shapes)
    x = BOARD_WIDTH // 2 - len(shape[0]) // 2
    return Block(x, 0, shape)


# 游戏循环


def game_loop():
    board = Board()
    current_block = generate_new_block()
    next_block = generate_new_block()
    score = 0
    while True:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == LEFT:
                    if board.is_valid_position(
                        Block(current_block.x - 1,
                              current_block.y, current_block.shape)
                    ):
                        current_block.move(-1, 0)
                elif event.key == RIGHT:
                    if board.is_valid_position(
                        Block(
                            current_block.x + 1,
                            current_block.y,
                            current_block.shape,
                        )
                    ):
                        current_block.move(1, 0)
                elif event.key == DOWN:
                    if board.is_valid_position(
                        Block(
                            current_block.x,
                            current_block.y + 1,
                            current_block.shape,
                        )
                    ):
                        current_block.move(0, 1)
                elif event.key == ROTATE:
                    rotated_block = Block(
                        current_block.x,
                        current_block.y,
                        current_block.shape,
                    )
                    rotated_block.rotate()
                    if board.is_valid_position(rotated_block):
                        current_block = rotated_block
                        # 移动方块
                    if board.is_valid_position(
                        Block(
                            current_block.x,
                            current_block.y + 1,
                            current_block.shape,
                        )
                    ):
                        current_block.move(0, 1)
            else:
                board.add_block(current_block)
                num_rows_removed = board.remove_filled_rows()
                score += num_rows_removed
                current_block = next_block
                next_block = generate_new_block()
            if not board.is_valid_position(current_block):
                # 游戏结束
                return score
            # 渲染游戏界面
            game_window.fill((255, 255, 255))
            board.draw()
            current_block.draw()
            pygame.display.update()
            # 游戏帧率
            clock.tick(5)


# 运行游戏
score = game_loop()
print("游戏结束，得分：", score)

# 退出 Pygame
pygame.quit()
