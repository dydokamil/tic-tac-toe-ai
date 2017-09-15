import numpy as np

from ttt_env import TicTacToeEnvironment

ACTIONS = 9
Q = []
EPISODES = 10000
ALPHA = .1
GAMMA = .99
D = []


def state_present(state):
    for s, a in Q:
        if np.all(s == state):
            return True

    return False


def add_state(state, Q_vals=None):
    if Q_vals is not None:
        Q.append([state, Q_vals])
    else:
        Q.append([state, np.zeros(9)])


def get_policy(state):
    for s, a in Q:
        if np.all(s == state):
            return a

    add_state(state)
    return np.arange(9)


def log_replay(s, a, r, s_prime, terminated):
    global D
    D.append((s, a, r, s_prime, terminated))


def get_best_action(state, environment):
    pi = get_policy(state)
    best_actions_sorted = np.argsort(pi)

    for a in best_actions_sorted:
        if environment.can_place_at(a):
            return a

    raise RuntimeError("PANIC!")


def remove_state(state):
    for i in range(len(Q)):
        if np.all(Q[i][0] == state):
            break

    del Q[i]


def update_policy(s, a, r, s_prime, terminated, step):
    if not state_present(s):
        add_state(s)

    if not state_present(s_prime):
        add_state(s_prime)

    pi = get_policy(s)
    pi_prime = get_policy(s_prime)
    max_a_prime = np.max(pi_prime)
    Q_sa = pi[a]

    remove_state(s)

    if terminated:
        pi[a] = r
    else:
        pi[a] += ALPHA * (r + GAMMA ** step * np.max(pi_prime))

    add_state(s, pi)


def q_learning_play():
    env = TicTacToeEnvironment()

    for i in range(EPISODES):
        s = env.reset()
        terminated = False
        step = 1
        while not terminated:
            if env.is_circles_turn():
                a = env.step_sample()
            else:
                a = get_best_action(s, env)

            s_prime, r, terminated = env.step(a)
            # log_replay(s, a, r, s_prime, terminated)
            update_policy(s, a, r, s_prime, terminated, step)
            # env.render()
            if not terminated:
                s = s_prime
            step += 1


def interactive_play():
    env = TicTacToeEnvironment()
    for i in range(EPISODES):
        s = env.reset()
        env.render()
        terminated = False
        step = 1
        while not terminated:
            if env.is_circles_turn():
                a = int(input("Choose your action (0-8): "))
            else:
                a = get_best_action(s, env)

            s_prime, r, terminated = env.step(a)
            env.render()
            if not terminated:
                s = s_prime
            step += 1


if __name__ == '__main__':
    # random_play()
    print("Training the model...")
    q_learning_play()
    print("Now you go!")
    interactive_play()
