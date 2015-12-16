

import pylab as plt

from retrieve_SkipGram_weights import plot_CG_chord2vec_weights, plot_SGD_chord2vec_weights
from retrieve_model_tools import retrieve_SkipGramNN, retrieve_SkipGramNN_SGD


def plot_SGD_embedding(highlight_syms=None, zoom=False):
    if highlight_syms is None:
        highlight_syms = ['C', 'F', 'G', 'Cm', 'B-', 'A-',
                          'Fmaj7', 'Fmaj7/A', 'Am']

    print '====== SGD ======'
    plt.figure()
    plot_SGD_chord2vec_weights(highlight_syms=highlight_syms,
                               zoom=zoom)


def highlight_syms_compare():
    highlight_syms = ['C', 'F', 'G', 'Cm', 'B-', 'A-',
                      'Fmaj7', 'Fmaj7/A', 'Am']

    plot_SGD_embedding(highlight_syms, zoom=True)

    print '====== CG ======'
    plt.figure()
    plot_CG_chord2vec_weights(highlight_syms=highlight_syms)


def query_neighborhood_compare(query_chord, topn, nn, nn_sgd):
    chords = nn.most_similar(query_chord, topn=topn, return_scores=False)
    print 'CG most similar chord for %s:' % query_chord, chords

    chords = nn_sgd.most_similar(query_chord, topn=topn, return_scores=False)
    print 'SGD most similar chord for %s:' % query_chord, chords
    print


def neighborhood_compare():
    topn = 7

    nn = retrieve_SkipGramNN()
    nn_sgd = retrieve_SkipGramNN_SGD()

    query_neighborhood_compare('Am', topn, nn, nn_sgd)
    query_neighborhood_compare('F', topn, nn, nn_sgd)
    query_neighborhood_compare('C', topn, nn, nn_sgd)
    query_neighborhood_compare('G', topn, nn, nn_sgd)


if __name__ == '__main__':
    # highlight_syms_compare()
    neighborhood_compare()

    # plot_SGD_embedding()