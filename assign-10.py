import numpy as np
import random

class TicTacToe:
    def __init__(self):
        self.board = np.zeros(9, dtype=int)

    def reset(self):
        self.board = np.zeros(9, dtype=int)
        return self.board.copy()

    def available_moves(self):
        return [i for i in range(9) if self.board[i] == 0]

    def play(self, move, player):
        if self.board[move] == 0:
            self.board[move] = player
            return True
        return False

    def winner(self):
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]

        for line in lines:
            if abs(sum(self.board[line])) == 3:
                return int(self.board[line[0]])

        if len(self.available_moves()) == 0:
            return 0

        return None

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, moves, training=True):
        if training and random.random() < self.epsilon:
            return random.choice(moves)

        values = [self.get_q(state, a) for a in moves]
        best = max(values)
        choices = [a for a, v in zip(moves, values) if v == best]
        return random.choice(choices)

    def update(self, state, action, reward, next_state, next_moves, done):
        old_q = self.get_q(state, action)

        if done or not next_moves:
            future = 0
        else:
            future = max(self.get_q(next_state, a) for a in next_moves)

        self.q_table[(state, action)] = old_q + self.alpha * (
            reward + self.gamma * future - old_q
        )

def train(agent, episodes=50000):
    game = TicTacToe()

    for _ in range(episodes):
        game.reset()
        state = tuple(game.board)
        history = []
        player = 1

        while True:
            moves = game.available_moves()

            if player == 1:
                action = agent.choose_action(state, moves)
                game.play(action, 1)
                next_state = tuple(game.board)
                history.append((state, action, next_state))
            else:
                action = random.choice(moves)
                game.play(action, -1)
                next_state = tuple(game.board)

            result = game.winner()

            if result is not None:
                if result == 1:
                    reward = 1
                elif result == -1:
                    reward = -1
                else:
                    reward = 0.2

                for s, a, ns in reversed(history):
                    agent.update(s, a, reward, ns, [], True)
                    reward *= agent.gamma

                break

            state = tuple(game.board)
            player *= -1

    agent.epsilon = 0

def test(agent, games=1000):
    results = {"Wins": 0, "Losses": 0, "Draws": 0}
    game = TicTacToe()

    for _ in range(games):
        game.reset()
        player = 1

        while True:
            moves = game.available_moves()

            if player == 1:
                state = tuple(game.board)
                action = agent.choose_action(state, moves, training=False)
                game.play(action, 1)
            else:
                action = random.choice(moves)
                game.play(action, -1)

            result = game.winner()

            if result is not None:
                if result == 1:
                    results["Wins"] += 1
                elif result == -1:
                    results["Losses"] += 1
                else:
                    results["Draws"] += 1
                break

            player *= -1

    print("Testing Results:", results)
    print("Win Rate:", round(results["Wins"] / games * 100, 2), "%")

agent = QLearningAgent()
train(agent)
test(agent)

game = TicTacToe()
game.reset()

print("\nTrained Agent Playing:")
while True:
    print(game.board.reshape(3, 3))

    moves = game.available_moves()
    state = tuple(game.board)
    action = agent.choose_action(state, moves, training=False)
    game.play(action, 1)

    result = game.winner()

    if result is not None:
        print(game.board.reshape(3, 3))
        print("Agent Result:", "Win" if result == 1 else "Draw" if result == 0 else "Loss")
        break

    action = random.choice(game.available_moves())
    game.play(action, -1)

    result = game.winner()

    if result is not None:
        print(game.board.reshape(3, 3))
        print("Agent Result:", "Win" if result == 1 else "Draw" if result == 0 else "Loss")
        break




# Reinforcement Learning- 
# Build a Tic-Tac-Toe game using reinforcement learning in Python by using following 
# tasks 
#  a. Setting up the environment.  
#  b. Defining the Tic-Tac-Toe game. 
#  c. Building the reinforcement learning model. 
#  d. Training the model.  
#  e. Testing the model. 
