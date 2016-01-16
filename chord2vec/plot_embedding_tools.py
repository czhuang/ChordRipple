
import os

from copy import copy
import cPickle as pickle

import numpy as np
import pylab as plt
from sklearn.decomposition import PCA

from music_theory_tools import CIRCLE_OF_FIFTHS_MAJOR, CIRCLE_OF_FIFTHS_MINOR
# from music_theory_tools import CIRCLE_OF_FIFTHS_MAJOR_ENHARM, CIRCLE_OF_FIFTHS_MINOR_ENHARM
from music_theory_tools import CIRCLE_OF_FIFTHS_MAJOR_DICT, CIRCLE_OF_FIFTHS_MINOR_DICT
# from music_theory_tools import CIRCLE_OF_FIFTHS_MAJOR_ENHARM_DICT, CIRCLE_OF_FIFTHS_MINOR_ENHARM_DICT
from plot_utilities import add_arrow_annotation, annotate, filter_syms
from config import get_configs


def plot_vec(vecs, syms, highlight_syms=None,
             highlight_only=False, doPCA=True,
             tag='', transposed_to_C=True, save=True, return_vecs=False, circles_later=True):

    if doPCA:
        pca = PCA(n_components=2)
        vecs = pca.fit_transform(vecs)
        print '=== PCA projection ==='
        print pca.explained_variance_ratio_
        print 'choosen explained: %.4f' % np.sum(pca.explained_variance_ratio_[:2])
    else:
        assert vecs.shape[1] == 2

    dot_size = 2
    plt.scatter(vecs[:, 0], vecs[:, 1], s=dot_size, color='b')

    ax = plt.gca()
    if not highlight_only:
        if np.size(vecs) > 0:
            filtered_vecs, filtered_syms = filter_syms(vecs, syms, exclude_syms=highlight_syms)
            annotate(filtered_syms, filtered_vecs, ax)

    if highlight_syms is not None:
        print 'highlight_syms:', highlight_syms
        print len(highlight_syms)
        filtered_highlight_vecs, filtered_highlight_syms = filter_syms(vecs, syms, include_syms=highlight_syms)

        # just for checking, which ones are not found?
        for sym in highlight_syms:
            if sym not in filtered_highlight_syms:
                print sym,
        print

        # assert len(highlight_syms) == len(highlight_vecs)
        highlight_dot_sizes = dot_size
        color = 'b'

        if np.size(filtered_highlight_vecs) > 0:
            plt.scatter(filtered_highlight_vecs[:, 0], filtered_highlight_vecs[:, 1],
                        s=highlight_dot_sizes, color=color)
            if not circles_later:
                annotate(filtered_highlight_syms, filtered_highlight_vecs, ax, color=color, text_size='large')

    title = "Chord2Vec embedding"
    if transposed_to_C:
        title += " (all songs transposed to C)"
    else:
        title += " (all songs kept in original key)"
    plt.title(title)
    if doPCA:
        plt.xlabel("first principal component")
        plt.ylabel("second principal component")

    # work out boundaries
    mins = np.min(vecs, axis=0)
    maxs = np.max(vecs, axis=0)

    # ranges = (mins[0], maxs[0], mins[1], maxs[1])
    padding = 0.15
    plt.xlim(mins[0]-padding, maxs[0]+padding)
    plt.ylim(mins[1]-padding, maxs[1]+padding)

    # draw vertical line
    plt.vlines(0, mins[1], maxs[1], linestyles='--', colors='b', linewidth=1)
    plt.hlines(0, mins[0], maxs[0], linestyles='--', colors='b', linewidth=1)

    plt.tick_params(axis='both', which='major', labelsize=6)
    if save:
        fname = '%s-%s.pdf' % ('chord_space', tag)
        print '...saving figure to', fname
        plt.savefig(fname)

    # plt.show()
    if return_vecs:
        return ax, vecs
    return ax


def zoom_vecs(vecs, syms, boundaries=None):
    # assumes incoming vecs are two dimensional
    assert vecs.shape[1] == 2
    if boundaries is None:
        # for x and y coordinates
        boundaries = [[-0.4, 0.4], [-0.4, 0.4]]
    print 'Initial number of chords to show:', vecs.shape[0]
    in_vecs = []
    in_syms = []
    out_vecs = []
    out_syms = []
    for i, xy in enumerate(vecs):
        if boundaries[0][0] < xy[0] < boundaries[0][1] and \
           boundaries[1][0] < xy[1] < boundaries[1][1]:
            in_vecs.append(xy)
            in_syms.append(syms[i])
        else:
            out_vecs.append(xy)
            out_syms.append(syms[i])
    print 'Zoomed in number of chords to show:', len(in_vecs)
    print 'Out of range chord syms:', out_syms
    return np.asarray(in_vecs), in_syms, np.asarray(out_vecs), out_syms


def plot_zoom_vec(vecs, syms, highlight_syms=None,
                  highlight_only=False, doPCA=True,
                  tag='', transposed_to_C=True):
    # To plot more zoomed in so that can see more granularity
    # for the vecs that have smaller norms

    # can only zoom in two dimensions for now
    assert doPCA
    assert np.size(vecs) > 0
    pca = PCA(n_components=2)
    vecs = pca.fit_transform(vecs)

    in_vecs, in_syms, out_vecs, out_syms = zoom_vecs(vecs, syms)
    assert in_vecs.shape[1] == 2

    in_highlight_syms = []
    for sym in highlight_syms:
        if sym in in_syms:
            in_highlight_syms.append(sym)

    plot_vec(in_vecs, in_syms, doPCA=False, highlight_only=highlight_only,
             highlight_syms=in_highlight_syms, tag=tag, transposed_to_C=transposed_to_C)


def replace_dash_with_b(syms):
    return [sym.replace('-', 'b') for sym in syms]


def plot_circles_from_SKipGram_SGD_pickle():
    # path = os.path.join('models', 'rock-letter', 'chord2vec')
    # fname = 'window-1_bigram-False_hiddenSize-20_crossEntropy-3.537_bestIter-19-maxEpoch-20_opt-SGD_l2reg-0.1000.pkl'
    # fpath = os.path.join(path, fname)
    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-20_crossEntropy-3.242_bestIter-9-maxEpoch-10_opt-SGD_l2reg-0.0100.pkl'
    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-20_crossEntropy-3.540_bestIter-19-maxEpoch-20_opt-SGD_l2reg-0.0010.pkl'
    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-20_crossEntropy-3.537_bestIter-19-maxEpoch-20_opt-SGD_l2reg-0.0010.pkl'
    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-20_crossEntropy-3.497_bestIter-29-maxEpoch-30_opt-SGD_l2reg-0.0010.pkl'
    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-20_crossEntropy-3.115_bestIter-26-maxEpoch-40_opt-SGD_l2reg-0.0010.pkl'
    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-20_crossEntropy-3.115_bestIter-24-maxEpoch-40_opt-SGD_l2reg-0.0010_batchSize-256.pkl'
    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-20_crossEntropy-3.128_bestIter-27-maxEpoch-40_opt-SGD_l2reg-0.0010_batchSize-256_momemtum-0.00.pkl'
    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-20_crossEntropy-3.132_bestIter-25-maxEpoch-40_opt-SGD_l2reg-0.0000_batchSize-256_momemtum-0.00.pkl'
    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-20_crossEntropy-3.127_bestIter-28-maxEpoch-40_opt-SGD_l2reg-0.0000_batchSize-256_momemtum-0.30.pkl'

    # with all the data
    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-20_crossEntropy-2.572_bestIter-39-maxEpoch-40_opt-SGD_l2reg-0.0000_batchSize-256_momemtum-0.30.pkl'
    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-20_crossEntropy-2.499_bestIter-54-maxEpoch-55_opt-SGD_l2reg-0.0000_batchSize-256_momemtum-0.00_N-11026_V-145.pkl'
    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-20_crossEntropy-2.469_bestIter-59-maxEpoch-60_opt-SGD_l2reg-0.0000_batchSize-256_momemtum-0.30_N-11026_V-145.pkl'
    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-20_crossEntropy-2.467_bestIter-59-maxEpoch-60_opt-SGD_l2reg-0.0000_batchSize-512_momemtum-0.30_N-11026_V-145.pkl'
    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-20_crossEntropy-2.480_bestIter-59-maxEpoch-60_opt-SGD_l2reg-0.0010_batchSize-512_momemtum-0.30_N-11026_V-145.pkl'
    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-20_crossEntropy-2.349_bestIter-99-maxEpoch-100_opt-SGD_l2reg-0.0005_batchSize-512_momemtum-0.30_N-11026_V-145.pkl'

    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-20_crossEntropy-2.216_bestIter-199-maxEpoch-200_opt-SGD_l2reg-0.0000_batchSize-512_momemtum-0.30_N-11026_V-145.pkl'

    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-30_crossEntropy-2.198_bestIter-199-maxEpoch-200_opt-SGD_l2reg-0.0000_batchSize-512_momemtum-0.30_N-11026_V-145.pkl'
    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-10_crossEntropy-2.292_bestIter-199-maxEpoch-200_opt-SGD_l2reg-0.0000_batchSize-512_momemtum-0.30_N-11026_V-145.pkl'
    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-20_crossEntropy-2.191_bestIter-249-maxEpoch-250_opt-SGD_l2reg-0.0000_batchSize-512_momemtum-0.15_N-11026_V-145.pkl'

    # the one used for paper
    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-20_crossEntropy-2.185_bestIter-248-maxEpoch-250_opt-SGD_l2reg-0.0000_batchSize-1024_momemtum-0.30_N-11026_V-145.pkl'

    # bach
    # not bigram, size 2
    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-2_crossEntropy-3.001_bestIter-249-maxEpoch-250_opt-SGD_l2reg-0.0000_batchSize-1024_momemtum-0.30_N-1155_V-93.pkl'

    # bach, not bigram, size 2, only the 30 chords, the 31st is a blank space
    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-2_crossEntropy-2.684_bestIter-99-maxIter-100_opt-SGD_l2reg-0.0100_batchSize-128_momemtum-0.30_N-950_V-31.pkl'

    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-5_crossEntropy-2.524_bestIter-99-maxIter-100_opt-SGD_l2reg-0.0100_batchSize-128_momemtum-0.30_N-950_V-31.pkl'

    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-2_crossEntropy-2.545_bestIter-199-maxIter-200_opt-SGD_l2reg-0.0100_batchSize-128_momemtum-0.30_N-950_V-31.pkl'
    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-2_crossEntropy-2.488_bestIter-399-maxIter-400_opt-SGD_l2reg-0.0001_batchSize-128_momemtum-0.30_N-950_V-31.pkl'
    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-2_crossEntropy-2.481_bestIter-499-maxIter-500_opt-SGD_l2reg-0.0001_batchSize-128_momemtum-0.30_N-950_V-31.pkl'

    W1, replaced_syms = read_SkipGram_pickle(fpath)
    plot_circles(W1, replaced_syms)


def plot_vecs_from_saved_vecs():
    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-10_crossEntropy-2.590_bestIter-29-maxIter-30_opt-SGD_l2reg-0.0001_batchSize-128_momemtum-0.30_N-10806_V-103.pkl'

    # transposed rock
    fpath = 'models/rock-letter/chord2vec/window-1_bigram-False_hiddenSize-10_crossEntropy-2.249_bestIter-29-maxIter-30_opt-SGD_l2reg-0.0001_batchSize-128_momemtum-0.30_N-11026_V-98.pkl'
    W1, replaced_syms = read_SkipGram_pickle(fpath)
    plot_vecs_wrapper(W1, replaced_syms)


def read_SkipGram_pickle(fpath, return_parser_weights=False):
    with open(fpath, 'rb') as p:
        results = pickle.load(p)
    parser = results['parser']
    W_vect = results['W']
    W1 = parser.get(W_vect, ('weights', 1))
    print 'W1 size:', W1.shape
    syms = results['syms']
    replaced_syms = replace_dash_with_b(syms)
    if return_parser_weights:
        return W1, replaced_syms, parser, W_vect
    return W1, replaced_syms


def plot_circles_from_SkipGramNN_CG_pickle():
    fname = 'w1-regularize_0-learn_rate_0.00-use_letternames_1-bigram_0-transposed_0-opt_algorithm_cg-max_iter_500-retrieve_model_0-window_1-duplicate_by_rotate_0-lay-2016-01-03_14-55-28.pkl'
    # with open(fname, 'wb') as p:
    #     pickle.dump(model.W1.value, p)
    #     pickle.dump(data.syms, p)
    #     pickle.dump(model_loss, p)
    #     pickle.dump(model_weights, p)
    #     pickle.dump(configs, p)

    with open(fname, 'rb') as p:
        W1 = pickle.load(p)
        syms = pickle.load(p)
    plot_circles(W1, syms)


def plot_circles(W1, syms):
    print '# of syms:', len(syms)
    highlight_syms = copy(CIRCLE_OF_FIFTHS_MAJOR)
    highlight_syms.extend(copy(CIRCLE_OF_FIFTHS_MINOR))
    # highlight_syms.extend(copy(CIRCLE_OF_FIFTHS_MAJOR_ENHARM))
    # highlight_syms.extend(copy(CIRCLE_OF_FIFTHS_MINOR_ENHARM))
    ax, vecs = plot_vec(W1, syms, highlight_syms=highlight_syms, tag='circle_test', save=False, return_vecs=True,
                        transposed_to_C=False)
    add_arrow_annotation(syms, vecs, CIRCLE_OF_FIFTHS_MAJOR_DICT, ax, color='g')
    add_arrow_annotation(syms, vecs, CIRCLE_OF_FIFTHS_MINOR_DICT, ax, color='m')

    # add_arrow_annotation(syms, vecs, CIRCLE_OF_FIFTHS_MAJOR_ENHARM_DICT, ax, color='g')
    # add_arrow_annotation(syms, vecs, CIRCLE_OF_FIFTHS_MINOR_ENHARM_DICT, ax, color='m')
    plt.tight_layout()

    id = np.random.random_integers(1000000)
    print 'id: %d' % id
    plt.savefig('circle_test-%d.pdf' % id)


def plot_vecs_wrapper(W1, syms):
    highlight_syms = None
    print '# of syms:', len(syms)
    ax, vecs = plot_vec(W1, syms, highlight_syms=highlight_syms, tag='vecs_test', save=False, return_vecs=True,
                        transposed_to_C=True)
    plt.tight_layout()

    id = np.random.random_integers(1000000)
    print 'id: %d' % id
    plt.savefig('circle_test-%d.pdf' % id)


def plot_axes():
    # need unigram distribution to plot only chords that have high count, top 30
    from make_model_tools import make_Ngram
    ngram = make_Ngram()
    top_syms = ngram.get_top_count_syms(topn=30)

    from plot_utilities import plot_mat_sorted_with_itself
    configs = get_configs()

    # chosen b/c the weights are less extreme, it seems the regularization worked out better here
    # SGD, with a little L2-regularization, min count = 5
    fpath = 'models/rock-letter/chord2vec/window-1_bigram-True_hiddenSize-1_crossEntropy-2.589_bestIter-999-maxIter-1000_opt-SGD_l2reg-0.0100_batchSize-256_momemtum-0.10_N-950_V-31.pkl'
    # --- U
    # ii6/5 iv6 ii/o6/5 IV7 VI I6/4 i6 i6/4 iio6 III ii7 ii V6/5/V I6 iv i v vi IV6 I IV iii V6 V V2 V4/3 viio6 vii/o7 V6/5 V7
    # --- V1
    # i6/4 I6/4 ii7 V iio6 iv ii/o6/5 v V6/5/V III IV7 viio6 ii6/5 iv6 vii/o7 V4/3 IV ii V6 iii IV6 V6/5 V7 V2 i6 vi I6 VI i I

    # # 'CG', min count = 5, no regularization
    # fpath = 'models/rock-letter/chord2vec/window-1_bigram-True_hiddenSize-1_crossEntropy-2.569_bestIter-837-maxIter-500_opt-CG_l2reg-0.0000_batchSize-32_momemtum-0.30_N-950_V-31.pkl'
    # # with regularzation
    # fpath = 'models/rock-letter/chord2vec/window-1_bigram-True_hiddenSize-1_crossEntropy-2.573_bestIter-808-maxIter-500_opt-CG_l2reg-0.0100_batchSize-128_momemtum-0.30_N-950_V-31.pkl'

    W1, replaced_syms, parser, W_vect = read_SkipGram_pickle(fpath, return_parser_weights=True)

    W1_vecs, W1_syms = filter_syms(W1, replaced_syms, include_syms=top_syms)
    row_tag = 'U'
    print '---', row_tag
    plot_mat_sorted_with_itself(W1_vecs.T, W1_syms, configs, row_tag, save=True)
    # plot_1d_weights(W1_vecs.T, W1_syms, 'W1')


    W2 = parser.get(W_vect, ('weights', 2, '1'))
    W2_vecs, W2_syms = filter_syms(W2.T, replaced_syms, include_syms=top_syms)
    row_tag = 'V'
    print '---', row_tag
    plot_mat_sorted_with_itself(W2_vecs.T, W2_syms, configs, row_tag, save=True)



def plot_1d_weights(weights, syms, tag):
    weights = np.squeeze(weights)
    # # weights = np.log(weights)
    # for weight in weights:
    #     print weight, np.log(weight)
    # need to sort first
    plt.figure()
    # first draw horizontal line, get range
    h_padding = np.std(weights) / 10
    left = np.max(weights) + h_padding
    right = np.max(weights) + h_padding
    # this axis line should be thicker
    plt.hlines(0, left, right, linewidths=5)

    tick_half_height = 20
    for weight in weights:
        plt.vlines(weight, -tick_half_height, tick_half_height)

    fname = 'test_1d_%s.pdf' % tag
    plt.savefig(fname)



if __name__ == '__main__':
    # plot_circles_from_SkipGramNN_CG_pickle()
    # plot_circles_from_SKipGram_SGD_pickle()
    # plot_axes()
    plot_vecs_from_saved_vecs()
