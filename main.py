import numpy as np

from ttt_env import TicTacToeEnvironment

ACTIONS = 9
Q = {}
EPISODES = 100000
ALPHA = .1
GAMMA = .5
D = []


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


def update_policy(s, a, r, s_prime, terminated):
    # TODO implement reward discount
    pi = get_policy(s)  # pi[a] == Q(s, a)
    pi_prime = get_policy(s_prime)
    max_a_prime = np.max(pi_prime)

    if terminated:
        pi[a] = r
    else:
        pi[a] += ALPHA * (r + max_a_prime - pi[a])

    Q[str(s)] = pi


def q_learning_play():
    env = TicTacToeEnvironment()

    for i in range(EPISODES):
        s = env.reset()
        terminated = False
        while not terminated:
            if env.is_circles_turn():
                _, _, terminated = env.step(env.step_sample())
            else:
                s = env.get_state()
                a = get_best_action(s, env)
                s_prime, r, terminated = env.step(a)
                update_policy(s, a, r, s_prime, terminated)


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
                if not state_present(s):
                    print("Guessing")
                else:
                    print(get_policy(s))
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
