import numpy as np

from ttt_env import TicTacToeEnvironment

ACTIONS = 9
Q = []
EPISODES = 5000


def random_play():
    grid = np.zeros((3, 3))
    circles_move = True

    while not check_win(grid):
        m = np.random.choice(np.where(grid == 0)[0])
        move(grid, m, circles_move)
        circles_move = not circles_move

    winner = check_win(grid)
    if check_win(grid) == 3:
        print("Tie!")
    else:
        print(winner, 'won.')
    print(grid)


def add_state(state, check=True):
    if check:

    Q.append([state, np.zeros(9)])
    return True


def get_policy(state):
    for s, a in Q:
        if s == state:
            return a

    add_state(state)
    return np.zeros(9)


def choose_action(grid, epsilon):
    if np.random.rand() <= epsilon:
    else:
        return get_policy(grid)


def update_policy(s, a, r, s_prime):
    pass


def q_learning_play():
    env = TicTacToeEnvironment()
    epsilon = 1.
    EPSILON_DECAY = .99

    for i in range(EPISODES):
        s = env.reset()
        terminated = False
        while not terminated:
            a = choose_action(s, epsilon)
            s_prime, r, terminated = env.step(a)
            if not terminated:
                s = s_prime

        epsilon *= EPSILON_DECAY
        update_policy(s, a, r, s_prime)

    while True:
        s = env.reset()
        while not check_win(grid):
            move(grid, choose_action(grid, epsilon), circles_move)
            circles_move = not circles_move


if __name__ == '__main__':
    # random_play()
    q_learning_play()
