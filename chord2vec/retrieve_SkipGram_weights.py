

import os
import cPickle as pickle

import numpy as np

from utility_tools import retrieve_most_recent_fname
from plot_embedding_tools import plot_vec, plot_zoom_vec


def retrieve_most_recent_pkl():
    fpath = os.path.join('models', 'rock-letter', 'chord2vec')
    fname = retrieve_most_recent_fname(fpath)

    with open(fname, 'rb') as p:
        results = pickle.load(p)

    parser = results['parser']
    print parser.idxs_and_shapes

    W_vect = results['W']
    syms = results['syms']

    # this is the first layer weights
    in_W = parser.get(W_vect, ('weights', 1))

    highlight_syms = ['C', 'F', 'G']
    tag = os.path.splitext(os.path.split(fname)[-1])[0]
    print 'tag:', tag
    plot_vec(in_W, syms, highlight_syms, tag=tag)

    print 'cross_entropy: %.4f' % results['cross_entropy']
    print 'best iteration: %d' % results['iter']


def retrieve_chord2vec_weights(most_recent=False, return_fname=False):
    # Chord2Vec embedding optimized with stochastic gradient descent (SGD)
    fpath = os.path.join('models', 'rock-letter',
                         'chord2vec')  #, 'best_models')
    if most_recent:
        fname = retrieve_most_recent_fname(fpath)
    else:
        fname = 'window-1_bigram-False_hiddenSize-20_crossEntropy-2.414_bestIter-79-maxEpoch-80_opt-SGD_l2reg-0.0100.pkl'

    with open(os.path.join(fpath, fname), 'rb') as p:
        results = pickle.load(p)

    parser = results['parser']
    print parser.idxs_and_shapes

    W_vect = results['W']
    syms = results['syms']
    # this is the first layer weights
    in_W = parser.get(W_vect, ('weights', 1))

    print 'cross_entropy: %.4f' % results['cross_entropy']
    print 'best iteration: %d' % results['iter']

    if return_fname:
        return in_W, syms, fname
    else:
        return in_W, syms


def plot_SGD_chord2vec_weights(most_recent=False, highlight_syms=None,
                               zoom=False):
    in_W, syms, fname = retrieve_chord2vec_weights(return_fname=True)
    in_W, syms, retrieve_chord2vec_weights(most_recent)
    if highlight_syms is None:
        highlight_syms = ['C', 'F', 'G']
    tag = os.path.splitext(os.path.split(fname)[-1])[0]
    print 'tag:', tag
    if not zoom:
        plot_vec(in_W, syms, highlight_syms=highlight_syms, tag=tag)
    else:
        plot_zoom_vec(in_W, syms, highlight_syms=highlight_syms, tag=tag+"_zoomed")


def plot_CG_chord2vec_weights(highlight_syms=None):
    # Chord2Vec embedding optimized with conjugate gradients (CG)
    from retrieve_model_tools import retrieve_SkipGramNN
    nn = retrieve_SkipGramNN()
    if highlight_syms is None:
        highlight_syms = ['C', 'F', 'G']
    tag = 'previously'
    plot_vec(nn.W1, nn.syms, highlight_syms=highlight_syms, tag=tag)


if __name__ == '__main__':
    retrieve_chord2vec_weights()
    # retrieve_previous_weights()



