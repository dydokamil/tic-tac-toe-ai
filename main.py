import os
import pickle

import numpy as np

from ttt_env import TicTacToeEnvironment

ACTIONS = 9
Q = {}
EPISODES = 100000
ALPHA = .1
GAMMA = .5
D = []
DUMP_FILE = 'qtable.p'


def state_present(state):
    if type(state) is not str:
        state = str(state)

    return state in Q


def add_state(state, Q_vals=None):
    if Q_vals is not None:
        Q[state] = Q_vals
    else:
        Q[state] = np.zeros(9)


def get_policy(state):
    state = str(state)
    if state_present(state):
        return Q[state]

    add_state(state)
    return np.zeros(9)


def get_best_action(state, environment):
    pi = get_policy(state)
    best_actions_sorted = np.argsort(pi)[::-1]

    for a in best_actions_sorted:
        if environment.can_place_at(a):
            return a

    raise RuntimeError("PANIC!")


def discount_rewards(r, length):
    discounted_r = np.zeros(length)
    for t in range(length):
        discounted_r[t] = r * GAMMA ** t
    return discounted_r[::-1]


def update_policy(r):
    global D
    dd = np.asarray(D)
    discounted = discount_rewards(r, len(dd))

    for (s, a, s_prime), r in zip(dd, discounted):
        pi = get_policy(s)  # pi[a] == Q(s, a)
        pi_prime = get_policy(s_prime)
        max_a_prime = np.max(pi_prime)

        pi[a] += ALPHA * (r + max_a_prime - pi[a])
        Q[str(s)] = pi

    D = []


def q_learning_play():
    env = TicTacToeEnvironment()

    for i in range(EPISODES):
        s = env.reset()
        terminated = False
        while not terminated:
            if env.is_circles_turn():
                s_prime, r, terminated = env.step(env.step_sample())
            else:
                s = env.get_state()
                a = get_best_action(s, env)
                s_prime, r, terminated = env.step(a)
                memorize_transition(s, a, s_prime)

        update_policy(r)
    save_q_table()


def save_q_table():
    print("Saving the model...")
    pkl = pickle.dump(Q, open(DUMP_FILE, 'wb'))


def memorize_transition(s, a, s_prime):
    D.append([s, a, s_prime])


def interactive_play():
    env = TicTacToeEnvironment()
    for i in range(EPISODES):
        print('*' * 10, "Game", i + 1, '*' * 10)
        s = env.reset()
        env.render()
        terminated = False
        step = 1
        while not terminated:
            if env.is_circles_turn():
                a = int(input("Choose your action (0-8): "))
            else:
                print("Thinking...")
                a = get_best_action(s, env)

            s_prime, r, terminated = env.step(a)
            env.render()
            if not terminated:
                s = s_prime
            step += 1
        if r == 5:
            print("You lost.")
        elif r == -5:
            print("You won.")
        elif r == -1:
            print("Draw.")


if __name__ == '__main__':
    if os.path.isfile(DUMP_FILE):
        print("Loading a saved model...")
        Q = pickle.load(open(DUMP_FILE, 'rb'))
    else:
        print("Training the model...")
        q_learning_play()

    print("Now you go!")
    interactive_play()
