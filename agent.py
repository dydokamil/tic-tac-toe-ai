import os
import pickle

import numpy as np


class Agent:
    def __init__(self, name):
        self.__Q__ = {}
        self.__name__ = name
        self.__dump_file__ = f'qtable_{self.__name__}.p'
        self.__D__ = []
        self.loaded = False
        self.__load_model__()

    def __load_model__(self):
        if os.path.isfile(self.__dump_file__):
            print(f"Loading a saved model for {self.__name__}...")
            self.__Q__ = pickle.load(open(self.__dump_file__, 'rb'))
            self.loaded = True

    def save_q_table(self):
        print(f"Saving the model for {self.__name__}...")
        pickle.dump(self.__Q__, open(self.__dump_file__, 'wb'))

    def memorize_transition(self, s, a, s_prime):
        self.__D__.append([s, a, s_prime])

    def __discount_rewards__(self, r, gamma):
        discounted_r = np.zeros(len(self.__D__))
        for t in range(len(self.__D__)):
            discounted_r[t] = r * gamma ** t
        return discounted_r[::-1]

    def __state_present__(self, state):
        if type(state) is not str:
            state = str(state)

        return state in self.__Q__

    def __get_policy__(self, state):
        state = str(state)
        if self.__state_present__(state):
            return self.__Q__[state]

        self.__add_state__(state)
        return np.zeros(9)

    def __add_state__(self, state, Q_vals=None):
        if Q_vals is not None:
            self.__Q__[state] = Q_vals
        else:
            self.__Q__[state] = np.zeros(9)

    def get_best_action(self, state, environment):
        pi = self.__get_policy__(state)
        best_actions_sorted = np.argsort(pi)[::-1]

        for a in best_actions_sorted:
            if environment.can_place_at(a):
                return a

        raise RuntimeError("PANIC!")

    def update_policy(self, r, gamma, alpha):
        dd = np.asarray(self.__D__)
        discounted = self.__discount_rewards__(r, gamma)

        for (s, a, s_prime), r in zip(dd, discounted):
            pi = self.__get_policy__(s)  # pi[a] == Q(s, a)
            pi_prime = self.__get_policy__(s_prime)
            max_a_prime = np.max(pi_prime)

            pi[a] += alpha * (r + max_a_prime - pi[a])
            self.__Q__[str(s)] = pi

        self.__D__ = []
