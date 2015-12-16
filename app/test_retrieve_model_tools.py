
import numpy as np

from retrieve_model_tools import retrieve_SkipGramNN
from utility_tools import print_vector, print_array


def check_main_chord_theta():
    nn = retrieve_SkipGramNN()
    syms = ['C', 'F', 'G', 'G7', 'Gsus4', 'A7', ]
    for sym in syms:
        print sym, nn.theta(sym) #, nn.theta(sym) * 180

    thetas = [nn.theta(sym) for sym in nn.syms]
    print_vector(thetas, 'thetas')

    sorted_indices = np.argsort(thetas)
    for idx in sorted_indices:
        print nn.syms[idx],
    print
    print len(nn.syms), len(list(set(nn.syms)))
    for idx in sorted_indices:
        print thetas[idx],
    print


def novelty_based_on_norm():
    # TODO: novelty_based_on_norm doesn't really work
    nn = retrieve_SkipGramNN()
    syms = nn.syms
    norms = [nn.norm(sym) for sym in syms]
    sorted_indices = np.argsort(norms)
    for i in sorted_indices:
        print syms[i], norms[i]


if __name__ == '__main__':
    # check_main_chord_theta()
    novelty_based_on_norm()
