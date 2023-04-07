import tensorflow as tf
import numpy as np
import pygame
import random
import os


# 初始化 Pygame
pygame.init()

#游戏action
ACTION_SIZE_GAME=4

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
ROTATE = pygame.K_UP

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
        self.color = 1

    def draw(self):
        leny = len(self.shape)
        for y in range(leny):
            lenx = len(self.shape[y])
            for x in range(lenx):
                if self.shape[y][x] != 0:
                    pass
                    pygame.draw.rect(
                        game_window,
                        self.color,
                        (
                            BOARD_POS_X + (self.x + x) * BLOCK_SIZE,
                            BOARD_POS_Y + (self.y + y) * BLOCK_SIZE,
                            BLOCK_SIZE,
                            BLOCK_SIZE,
                        ),
                    )

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def rotate(self):
        # 转置
        rotated = list(map(list, zip(*self.shape)))
        # 翻转每一行
        rotated = [row[::-1] for row in rotated]
        self.shape = rotated



# 定义游戏板类
class Board:
    def __init__(self):
        self.board = [[0 for _ in range(BOARD_WIDTH)]
                      for _ in range(BOARD_HEIGHT)]
        self.current_block = 1
        self.next_block = 1
        self.game_over = 0 

    def get_state(self):
        return np.array(self.board)

    def do_action(self, action):
        if action == 1:
            if self.is_valid_position(Block(self.current_block.x - 1, self.current_block.y, self.current_block.shape)):
                self.current_block.move(-1, 0)
            return 1
        elif action == 2:
            if self.is_valid_position(Block(self.current_block.x + 1, self.current_block.y, self.current_block.shape)):
                self.current_block.move(1, 0)
            return 1
        elif action == 3:
            if self.is_valid_position(Block(self.current_block.x, self.current_block.y + 1, self.current_block.shape)):
                self.current_block.move(0, 1)
            return 1
        elif action == 0:
            rotated_block = Block(self.current_block.x, self.current_block.y, self.current_block.shape)
            rotated_block.rotate()
            if self.is_valid_position(rotated_block):
                self.current_block = rotated_block
            return 1


    def draw(self):
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if self.board[y][x]:
                    pass
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
        leny = len(block.shape)
        for y in range(leny):
            lenx = len(block.shape[y])
            for x in range(lenx):
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
        leny = len(block.shape)
        for y in range(leny):
            lenx = len(block.shape[y])
            for x in range(lenx):
                if block.shape[y][x] != 0:
                    self.board[block.y + y][block.x + x] = block.color

    def remove_filled_rows(self):
        num_rows_removed = 0
        y = BOARD_HEIGHT - 1
        while y >= 0:
            if 0 not in self.board[y]:
                num_rows_removed += 1
                for yy in range(y, 0, -1):
                    for x in range(BOARD_WIDTH):
                        self.board[yy][x] = self.board[yy - 1][x]
                for x in range(BOARD_WIDTH):
                    self.board[0][x] = 0
            else:
                y -= 1
            return num_rows_removed

    def is_game_over(self):
        return self.game_over

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


board = Board() # 创建棋盘
# 游戏循环
def game_loop():
    global board
    board = Board() # 创建棋盘
    board.current_block = generate_new_block() # 生成新方块
    next_block = generate_new_block() # 生成下一个方块
    score = 0 # 初始化得分
    
    while True:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    if board.is_valid_position(Block(board.current_block.x - 1, board.current_block.y, board.current_block.shape)):
                        board.current_block.move(-1, 0)
                elif event.key == pygame.K_RIGHT:
                    if board.is_valid_position(Block(board.current_block.x + 1, board.current_block.y, board.current_block.shape)):
                        board.current_block.move(1, 0)
                elif event.key == pygame.K_DOWN:
                    if board.is_valid_position(Block(board.current_block.x, board.current_block.y + 1, board.current_block.shape)):
                        board.current_block.move(0, 1)
                elif event.key == pygame.K_UP:
                    rotated_block = Block(board.current_block.x, board.current_block.y, board.current_block.shape)
                    rotated_block.rotate()
                    if board.is_valid_position(rotated_block):
                        board.current_block = rotated_block
    #     print("进行中##################：")
        # 移动方块
        if board.is_valid_position(Block(board.current_block.x, board.current_block.y + 1, board.current_block.shape)):
            board.current_block.move(0, 1)
        else:
            board.add_block(board.current_block)
            num_rows_removed = board.remove_filled_rows()
            score += num_rows_removed
            board.current_block = next_block
            next_block = generate_new_block()
            if not board.is_valid_position(board.current_block):
                # 游戏结束
                board.game_over = 1
                print("游戏结束，得分：", score)
                return score

        # 渲染游戏界面
        game_window.fill((255, 255, 255))
        board.draw()
        board.current_block.draw()

         #游戏帧率
        clock.tick(100)


# 定义状态空间、行动空间和奖励函数
STATE_SIZE = 10
ACTION_SIZE = 4
REWARD_FACTOR = 200

# 定义神经网络
def build_network():

    if os.path.exists('model.h5'):
      model = tf.keras.models.load_model('model.h5')
    else:
      model = tf.keras.models.Sequential([
          tf.keras.layers.Dense(128, activation='relu', input_shape=(STATE_SIZE,)),
          tf.keras.layers.Dense(64, activation='relu'),
          tf.keras.layers.Dense(ACTION_SIZE, activation='linear')
      ])
      model.compile(optimizer='adam', loss='mse')
    return model

# 定义经验回放器
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

# 定义强化学习代理
class DQNAgent:
    def __init__(self):
        self.model = build_network()
        self.target_model = build_network()
        self.memory = ReplayMemory(2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def act(self, state):
        #q_values = self.model.predict(state)
        #print(q_values[0])
        #print(np.argmax(q_values[0]))
        #return np.argmax(q_values[0])
        if np.random.rand() <= self.epsilon:
            return np.random.choice(range(ACTION_SIZE))
        else:
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])
    
    def replay(self, batch_size):
        minibatch = self.memory.sample(batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                q_future = max(self.target_model.predict(next_state)[0])
                target[0][action] = reward + q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay*self.epsilon)

    def save_model(self):
        tf.keras.models.save_model(self.model, 'model.h5')

# 定义游戏循环
agent = DQNAgent()
batch_size = 4
scores = []
epis = 100
score = 0

for episode in range(epis):
    print("####################################################")
    board = Board() # 创建棋盘
    board.current_block = generate_new_block() # 生成新方块
    next_block = generate_new_block() # 生成下一个方块
    state = board.get_state()
    score = 0
    trueScore = 0
    while True:
    
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    if board.is_valid_position(Block(board.current_block.x - 1, board.current_block.y, board.current_block.shape)):
                        board.current_block.move(-1, 0)
                        action=1
                elif event.key == pygame.K_RIGHT:
                    if board.is_valid_position(Block(board.current_block.x + 1, board.current_block.y, board.current_block.shape)):
                        board.current_block.move(1, 0)
                        action=2
                elif event.key == pygame.K_DOWN:
                    if board.is_valid_position(Block(board.current_block.x, board.current_block.y + 1, board.current_block.shape)):
                        board.current_block.move(0, 1)
                        action=3
                elif event.key == pygame.K_UP:
                    rotated_block = Block(board.current_block.x, board.current_block.y, board.current_block.shape)
                    rotated_block.rotate()
                    if board.is_valid_position(rotated_block):
                        board.current_block = rotated_block
                        action=0

        action = agent.act(state)
        reward = board.do_action(action)
        #动作后更新界面
        if board.is_valid_position(Block(board.current_block.x, board.current_block.y + 1, board.current_block.shape)):
          board.current_block.move(0, 1)
        else:
          board.add_block(board.current_block)
          num_rows_removed = board.remove_filled_rows()
          score += num_rows_removed * 100
          trueScore += num_rows_removed * 100
          board.current_block = next_block
          next_block = generate_new_block()
          if not board.is_valid_position(board.current_block):
              # 游戏结束
              board.game_over = 1
              print("游戏结束，得分：", trueScore)

        game_window.fill((255, 255, 255))
        board.draw()
        board.current_block.draw()
        pygame.display.update()
        
        #clock.tick(10)

        next_state = board.get_state()
        done = board.is_game_over()

        if done:
            reward = -REWARD_FACTOR
        if action ==1 or action==2 or action==3 or action==0:
            agent.memory.push(state, action, reward, next_state, done)
        state = next_state
        score += reward

        if done:
            agent.update_target_model()
            scores.append(score)
            print("Episode: {}/{}, Score: {}, Epsilon: {:.2}".format(episode+1, epis, score, agent.epsilon))
            break
        if len(agent.memory.memory) > batch_size:
            agent.replay(batch_size)
    agent.update_epsilon()
    agent.save_model()#每批次保存一下

