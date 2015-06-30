
from load_model_tools import get_models
from dynamic_programming_tools import *


def test_shortest_path():
    nn, ngram = get_models()
    syms = nn.data.syms
    # trans = nn.Ys['1'].value
    trans = ngram.ngram
    fixed = {0: 'I', 2: 'V', 4: 'I'}
    shortest_path(trans, syms, fixed)


def test_forward_backward():
    from retrieve_model_tools import retrieve_NGram
    ngram = retrieve_NGram()
    sorted_probs, sorted_syms = simple_foward_backward_gap_dist(ngram, 'I', 'I')
    for i in range(10):
        print sorted_probs[i], sorted_syms[i]


def test_continue_to_end():
    nn, ngram = get_models()
    # fixed = {0: 'I', 2: 'V', 7: 'I'}
    # original_seq = ['I', 'IV', 'V']

    fixed = {0: 'C', 2: 'G', 7: 'C'}
    original_seq = ['C', 'F', 'G']
    new_seq, sym_inds = shortest_path(nn, fixed, 2, original_seq)
    print new_seq
    print sym_inds


if __name__ == "__main__":
    # test_shortest_path()
    # test_forward_backward()
    test_continue_to_end()
