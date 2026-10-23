import numpy as np
import random

maze = [
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
]

rows = len(maze)
cols = len(maze[0])
start = (0, 0)
goal = (4, 4)

actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
arrows = ["^", "v", "<", ">"]

q_table = np.zeros((rows, cols, len(actions)))

alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.99
min_epsilon = 0.05
episodes = 500
max_steps = 100

def step(state, action):
    r = state[0] + actions[action][0]
    c = state[1] + actions[action][1]
    if r < 0 or r >= rows or c < 0 or c >= cols or maze[r][c] == 1:
        return state, -5, False
    if (r, c) == goal:
        return (r, c), 100, True
    return (r, c), -1, False

print("MAZE (S = start, G = goal, # = wall, . = path)")
for i in range(rows):
    line = ""
    for j in range(cols):
        if (i, j) == start:
            line += "S "
        elif (i, j) == goal:
            line += "G "
        elif maze[i][j] == 1:
            line += "# "
        else:
            line += ". "
    print(line)

print()
print("TRAINING")
reward_history = []
for episode in range(1, episodes + 1):
    state = start
    total_reward = 0
    for _ in range(max_steps):
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = int(np.argmax(q_table[state[0], state[1]]))
        next_state, reward, done = step(state, action)
        old_value = q_table[state[0], state[1], action]
        best_next = np.max(q_table[next_state[0], next_state[1]])
        q_table[state[0], state[1], action] = old_value + alpha * (reward + gamma * best_next - old_value)
        state = next_state
        total_reward += reward
        if done:
            break
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    reward_history.append(total_reward)
    if episode % 100 == 0:
        avg = np.mean(reward_history[-100:])
        print("Episode", episode, "| Average reward (last 100):", round(avg, 2), "| Epsilon:", round(epsilon, 3))

print()
print("LEARNED POLICY (best action in each cell)")
for i in range(rows):
    line = ""
    for j in range(cols):
        if maze[i][j] == 1:
            line += "# "
        elif (i, j) == goal:
            line += "G "
        else:
            line += arrows[int(np.argmax(q_table[i, j]))] + " "
    print(line)

print()
print("FINAL PATH FROM START TO GOAL")
state = start
path = [state]
steps = 0
while state != goal and steps < 50:
    action = int(np.argmax(q_table[state[0], state[1]]))
    state, _, _ = step(state, action)
    path.append(state)
    steps += 1

print("Path:", path)
print("Number of steps:", steps)
if state == goal:
    print("The agent reached the goal.")
else:
    print("The agent did not reach the goal. Increase the episodes.")
