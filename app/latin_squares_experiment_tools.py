__author__ = 'czhuang'


import os
import cPickle as pickle

import numpy as np


def is_valid_latin_square(grid):
    n_row, n_col = grid.shape
    assert n_row % n_col == 0
    quotient = n_row / n_col
    sum_of_elements = np.sum(np.arange(n_col))
    sum_of_col_wanted = quotient * sum_of_elements
    sum_of_cols = np.sum(grid, axis=0)
    # print sum_of_cols, sum_of_col_wanted
    if np.allclose(sum_of_cols, sum_of_col_wanted):
        return True
    else:
        return False


def gen_latin_squares(num_conditions, num_participants):
    num_tries = 100
    grid = np.zeros((num_participants, num_conditions))
    for j in range(num_tries):
        # print j
        for i in range(num_participants):
            grid[i, :] = np.random.permutation(num_conditions)
        if is_valid_latin_square(grid):
            return grid
    return None


def stack_condition_ordering(num_conditions, num_participants):
    quotient = num_participants / num_conditions
    ordering = None
    for i in range(quotient):
        local_ordering = gen_latin_squares(3, 3)
        if ordering is None:
            ordering = local_ordering
        else:
            ordering = np.vstack((ordering, local_ordering))

    duplicated_ordering = np.zeros_like(ordering)
    duplicated_ordering = np.vstack((duplicated_ordering, duplicated_ordering))
    for i in range(ordering.shape[0]):
        duplicated_ordering[i*2, :] = ordering[i, :]
        duplicated_ordering[i*2+1, :] = ordering[i, :]
    print duplicated_ordering

    assert np.allclose(np.sum(duplicated_ordering, axis=0),
                       quotient * 2 * np.sum(np.arange(num_conditions)))



    fpath = os.path.join('pkls', 'condition_ordering.pkl')
    print 'fpath', fpath
    with open(fpath, 'wb') as p:
        pickle.dump(duplicated_ordering, p)


def get_condition_ordering():
    fpath = os.path.join('pkls', 'condition_ordering.pkl')
    with open(fpath, 'rb') as p:
        grid = pickle.load(p)
    return grid




if __name__ == '__main__':
    print os.getcwd()
    condition_orderings = stack_condition_ordering(3, 3*10)
    condition_orderings = get_condition_ordering()
    print condition_orderings
    print condition_orderings.shape

