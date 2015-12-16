

import pylab as plt

from NGram import *
from load_model_tools import make_test_NGram
from retrieve_model_tools import retrieve_NGram


def test_make_model():
    ngram = make_test_NGram()
    syms = ngram.gen_seq(5)
    print syms
    syms = ngram.gen_continuation(syms, 5)
    print syms


def plot_mat(mat, title, syms, x_tick_syms=None):
    mat = np.asarray(mat)
    plt.matshow(mat)
    # plt.title(title)
    from pylab import setp
    if mat.shape[1] < 23:
        fontsize = 'medium'
    else:
        fontsize = 'xx-small'
    fontsize = 'small'
    if x_tick_syms is None:
        x_tick_syms = syms
    plt.yticks(range(len(x_tick_syms)), x_tick_syms)
    plt.xticks(range(len(syms)), syms)
    setp(plt.gca().get_yticklabels(), fontsize=fontsize)
    setp(plt.gca().get_xticklabels(), fontsize=fontsize)
    plt.title(title)
    # plt.colorbar(shrink=.8)
    plt.colorbar()


def test_plot_one_gram_all():
    # configs, data = get_configs_data()
    
    ngram = retrieve_NGram()
    unigram = ngram.unigram_counts
    sorted_inds = np.argsort(-unigram)
    sorted_unigram = [unigram[ind] for ind in sorted_inds]
    PLOT_LOG = False
    if PLOT_LOG:
        # TODO: should be log-log scale to get straightline
        plt.plot(np.log(sorted_unigram))
        plt.ylabel('log counts')
    else:
        plt.plot(sorted_unigram)
    plt.title('Bach chorale (size of data: %d)' % np.sum(unigram))
    # plot_mat(, 'bach', ngram.syms)
    # plot_one_gram(ngram.syms, ngram.seqs)
    plt.savefig('bach_chorale-chord-dist.pdf')


def test_next_top_n():
    model = retrieve_NGram()
    # 'Em9' does not exist
    # 'Fdim' does not
    query_sym = model.syms[0]
    syms = model.next_top_n(query_sym)
    print 'what are the most common symbols that follow %s' % query_sym
    print syms


def top_start_chords():
    model = retrieve_NGram()
    syms = model.get_top_start_chords(5)
    print syms


def test_unigram_novelty():
    model = retrieve_NGram()
    inv_freq = model.unigram_inverse_freq('C')
    idx = model.get_idx('C')
    print 'counts', model.unigram_counts[idx]
    print inv_freq


if __name__ == "__main__":
    # test_make_model()
    # test_next_top_n()
    # top_start_chords()
    test_unigram_novelty()
