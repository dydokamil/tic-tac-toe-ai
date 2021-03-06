import numpy as np


class TicTacToeEnvironment:
    def __init__(self, random_start=False):
        self.__grid__ = np.zeros((3, 3))
        self.__circles_turn__ = True
        self.__needs_reset = True
        self.__random_start__ = random_start

    def step(self, action):
        """Draw a circle/cross at @action index.
        Returns [state, reward, terminated]"""
        if self.__needs_reset:
            print("Call reset() first. Returning...")
            return

        grid_flattened = self.__grid__.ravel()
        if grid_flattened[action] != 0:
            raise ValueError("Action not allowed.")

        grid_flattened[action] = 1 if self.__circles_turn__ else 2
        self.__circles_turn__ = not self.__circles_turn__
        self.__grid__ = grid_flattened.reshape((3, 3))

        winner = self.__check_win__()
        grid_copy = np.copy(self.__grid__)
        if not winner:
            transition = [grid_copy, 0, False]
        elif winner == 1:
            transition = [grid_copy, -5, True]
        elif winner == 2:
            transition = [grid_copy, 5, True]
        elif winner == 3:
            transition = [grid_copy, -1, True]

        if transition[2] is True:
            self.__needs_reset = True

        return transition

    def step_sample(self):
        return np.random.choice(np.where(self.__grid__.flatten() == 0)[0])

    def get_state(self):
        return np.copy(self.__grid__)

    def can_place_at(self, idx):
        if self.__grid__.flatten()[idx] == 0:
            return True
        else:
            return False

    def is_circles_turn(self):
        return self.__circles_turn__

    def reset(self):
        self.__init__()
        self.__needs_reset = False
        self.__circles_turn__ = bool(np.random.randint(2)) \
            if self.__random_start__ \
            else True
        return np.copy(self.__grid__)

    def render(self):
        circles_squares = np.copy(self.__grid__).astype(str)
        circles_squares[np.where(circles_squares == '0.0')] = '#'
        circles_squares[np.where(circles_squares == '1.0')] = 'o'
        circles_squares[np.where(circles_squares == '2.0')] = 'x'
        for row in circles_squares:
            for element in row:
                print(element, ' ', end='')
            print()

    def __check_win__(self):
        if np.any(np.all(self.__grid__ == 1, axis=1)) or \
                np.any(np.all(self.__grid__ == 1, axis=0)) or \
                np.any(np.all(np.diag(self.__grid__) == 1)) or \
                np.any(np.all(np.diag(self.__grid__[::-1] == 1))):
            return 1

        if np.any(np.all(self.__grid__ == 2, axis=1)) or \
                np.any(np.all(self.__grid__ == 2, axis=0)) or \
                np.any(np.all(np.diag(self.__grid__) == 2)) or \
                np.any(np.all(np.diag(self.__grid__[::-1] == 2))):
            return 2

        # tie
        if not np.any(self.__grid__ == 0):
            return 3

        return None
