import numpy as np


def check_win(grid):
    if np.any(np.all(grid == 1, axis=1)) or \
            np.any(np.all(grid == 1, axis=0)) or \
            np.any(np.all(np.diag(grid) == 1)) or \
            np.any(np.all(np.diag(grid[::-1] == 1))):
        return 1

    if np.any(np.all(grid == 2, axis=1)) or \
            np.any(np.all(grid == 2, axis=0)) or \
            np.any(np.all(np.diag(grid) == 2)) or \
            np.any(np.all(np.diag(grid[::-1] == 2))):
        return 2

    # tie
    if not np.any(grid == 0):
        return 3

    return None


def move(grid, cell, circle):
    grid = grid.ravel()
    if grid[cell] != 0:
        return False

    grid[cell] = 1 if circle else 2
    return True


if __name__ == '__main__':
    grid = np.zeros((3, 3))
    circles_move = True

    while not check_win(grid):
        print()
        while not move(grid, np.random.randint(0, 9), circles_move):
            pass
        print(grid)
        circles_move = not circles_move

    print(check_win(grid), 'won')
    print(grid)
