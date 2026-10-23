import random

alpha = 0.5
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.9995
min_epsilon = 0.05
episodes = 20000
test_games = 1000

class TicTacToe:
    def __init__(self):
        self.board = [" "] * 9

    def get_state(self):
        return "".join(self.board)

    def available_moves(self):
        return [i for i in range(9) if self.board[i] == " "]

    def make_move(self, position, player):
        self.board[position] = player

    def check_winner(self):
        lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                 (0, 3, 6), (1, 4, 7), (2, 5, 8),
                 (0, 4, 8), (2, 4, 6)]
        for a, b, c in lines:
            if self.board[a] != " " and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        if " " not in self.board:
            return "Draw"
        return None

    def show(self):
        for i in range(0, 9, 3):
            print(" " + " | ".join(self.board[i:i + 3]))
            if i < 6:
                print("---+---+---")

class QAgent:
    def __init__(self):
        self.q = {}
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.q.get((state, action), 0.0)

    def choose_action(self, state, moves, explore=True):
        if explore and random.random() < self.epsilon:
            return random.choice(moves)
        values = [self.get_q(state, m) for m in moves]
        best = max(values)
        best_moves = [m for m, v in zip(moves, values) if v == best]
        return random.choice(best_moves)

    def update(self, state, action, reward, next_state, next_moves, done):
        if done:
            target = reward
        else:
            target = reward + gamma * max(self.get_q(next_state, m) for m in next_moves)
        old = self.get_q(state, action)
        self.q[(state, action)] = old + alpha * (target - old)

def play_game(agent, train=True):
    game = TicTacToe()
    if random.random() < 0.5:
        game.make_move(random.choice(game.available_moves()), "O")
    while True:
        state = game.get_state()
        action = agent.choose_action(state, game.available_moves(), explore=train)
        game.make_move(action, "X")
        result = game.check_winner()
        done = True
        if result == "X":
            reward = 1
        elif result == "Draw":
            reward = 0.5
        else:
            game.make_move(random.choice(game.available_moves()), "O")
            result = game.check_winner()
            if result == "O":
                reward = -1
            elif result == "Draw":
                reward = 0.5
            else:
                reward = 0
                done = False
        if train:
            agent.update(state, action, reward, game.get_state(), game.available_moves(), done)
        if done:
            return result

def test_agent(agent, games):
    wins = draws = losses = 0
    for _ in range(games):
        result = play_game(agent, train=False)
        if result == "X":
            wins += 1
        elif result == "Draw":
            draws += 1
        else:
            losses += 1
    return wins, draws, losses

def print_result(title, wins, draws, losses, games):
    print(title)
    print("Wins:", wins, "| Draws:", draws, "| Losses:", losses)
    print("Win rate:", round(wins / games * 100, 1), "%")
    print("Win + Draw rate:", round((wins + draws) / games * 100, 1), "%")

agent = QAgent()

print("TIC-TAC-TOE USING Q-LEARNING")
print("Agent plays X, opponent plays O (random moves)")
print()

wins, draws, losses = test_agent(agent, test_games)
print_result("BEFORE TRAINING (" + str(test_games) + " games)", wins, draws, losses, test_games)

print()
print("TRAINING FOR", episodes, "EPISODES")
recent = []
for episode in range(1, episodes + 1):
    result = play_game(agent, train=True)
    recent.append(result)
    agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)
    if episode % 5000 == 0:
        last = recent[-5000:]
        print("Episode", episode, "| Win rate (last 5000):", round(last.count("X") / len(last) * 100, 1), "% | Epsilon:", round(agent.epsilon, 3), "| States learned:", len(agent.q))

print()
wins, draws, losses = test_agent(agent, test_games)
print_result("AFTER TRAINING (" + str(test_games) + " games)", wins, draws, losses, test_games)

print()
print("SAMPLE GAME (trained agent = X, random opponent = O)")
game = TicTacToe()
while game.check_winner() is None:
    state = game.get_state()
    game.make_move(agent.choose_action(state, game.available_moves(), explore=False), "X")
    if game.check_winner() is None:
        game.make_move(random.choice(game.available_moves()), "O")
game.show()
print("Result:", game.check_winner())
