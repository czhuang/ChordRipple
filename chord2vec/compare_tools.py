

import pylab as plt
import numpy as np

from config import get_configs
from load_songs_tools import get_data
from load_model_tools import get_models
from PreprocessedData import PreprocessedSeq
from plot_utilities import plot_mat, plot_mat_sort_with, plot_mat_pca_ordered


def predict_probs_seq():
    # want for each sym a prediction list of prob for all syms
    ON_LOG = True
    # seq = ['I', 'IV', 'I6', 'V']
    # seq = [u'I', u'vi', u'V7', u'I', u'V2', u'I6', u'V', u'V7', u'I', u'V6/vi', u'V6/5/vi', u'IV6', u'vii/o7', u'I',
    #        u'IV7', u'V2', u'I', u'V', u'V7', u'I', u'V', u'V', u'viio6', u'i', u'vii/o7/V', u'V', u'V7', u'i', u'i',
    #        u'V4/3', u'i6', u'V6/5/iv', u'iv', u'V6/5/V', u'V', u'V7', u'i', u'V', u'I6', u'I', u'IV6', u'viio', u'I',
    #        u'vi', u'V6/5/V', u'V7/V', u'V', u'I6', u'IV', u'V7', u'IV6/5', u'viio', u'I', u'ii6/5', u'V']
    seq = [u'IV6', u'vii/o7', u'I', u'IV7']
    model, ngram_model = get_models()
    processed_seq = PreprocessedSeq(seq, model.data.syms, model.data.window)
    probs, mean_score_total, probs_all = model.predict_reverse([processed_seq],
                                                               return_probs=True,
                                                               return_all=True)
    ngram_probs, ngram_probs_all = ngram_model.predict(seq, log=False,
                                                       return_all=True)
    assert len(seq) == len(ngram_probs_all)
    print 'len(probs)', len(probs)
    print 'len()', len(probs[0])
    print len(probs_all[0][0])
    print len(probs_all[0][0])
    print len(model.data.syms)

    syms = model.data.syms
    topn = 8
    # b/c assumes multiple sequences
    probs_all = probs_all[0]
    for i, sym in enumerate(seq):
        probs = probs_all[i]
        inds = np.argsort(probs)[::-1]

        ngram_probs = ngram_probs_all[i]
        ngram_inds = np.argsort(ngram_probs)[::-1]
        print '--- %s ---' % sym
        for j in range(topn):
            # ind for nn
            ind = inds[j]
            ngram_ind = ngram_inds[j]

            if not ON_LOG:
                nn_prob = np.exp(probs[ind])
                ngram_prob = ngram_probs[ngram_ind]
            else:
                nn_prob = probs[ind]
                ngram_prob = np.log(ngram_probs[ngram_ind])

            print '%s:  \t%.2f, \t%s: \5%.4f' % (syms[ind],
                                                 nn_prob,
                                                 syms[ngram_ind],
                                                 ngram_prob)

    # ngram_probs = ngram_model.predict(seq)


def get_trans():
    configs = get_configs()
    data = get_data()
    nn, ngram = get_models()
    # if bigram nn, it's from t to t-1, then need to transpose?
    nn_trans_dict = nn.predict_identity()
    nn_trans = nn_trans_dict.values()[0]
    print '...nn_trans type', type(nn_trans), nn_trans.shape
    nn_trans = nn_trans_dict['1']
    print nn_trans_dict.keys()
    print '...nn_trans type', type(nn_trans), nn_trans.shape
    nn_trans = np.exp(nn_trans)
    ngram_trans = ngram.ngram
    return ngram_trans, nn_trans, ngram, nn, configs, data


def compare_transition_matrices():
    ngram_trans, nn_trans, ngram, nn, configs, data = get_trans()

    vec_to_sort_with = \
        plot_mat_pca_ordered(ngram_trans, data, configs,
                             fname_tag='ngram_trans')

    plot_mat_sort_with(vec_to_sort_with, data, configs,
                       nn_trans,
                       fname_tag='nn_trans')

    plot_mat_sort_with(vec_to_sort_with, data, configs,
                       np.log(ngram_trans),
                       fname_tag='ngram_trans-log-trans')


def query_syms_on_mat(query_syms, trans_mat, syms, configs, topn=7):
    trans_subset = []
    top_inds = []
    # collect top inds
    for i, sym in enumerate(query_syms):
        ind = syms.index(sym)
        trans = trans_mat[ind, :]
        sorted_inds = np.argsort(-trans)[:topn]
        print 'sorted_inds', sorted_inds
        print np.argsort(-trans)
        for ii in sorted_inds:
            top_inds.append(ii)
    top_inds = sorted(set(top_inds), key=top_inds.index)

    for i, sym in enumerate(query_syms):
        ind = syms.index(sym)
        trans = [trans_mat[ind, ii] for ii in top_inds]
        trans_subset.append(trans)
    top_syms = [syms[ii] for ii in top_inds]
    trans_subset = np.asarray(trans_subset)
    print 'before trans_subset', trans_subset.shape
    if len(trans_subset.shape) > 2:
        trans_subset = np.squeeze(trans_subset)
    print 'trans_subset', trans_subset.shape

    plot_mat(trans_subset, '',
             top_syms, x_tick_syms=query_syms)
    fname_tag = '_'.join(query_syms)
    fname_tag = fname_tag.replace('/', '_')
    plt.savefig('trans-%s-%s.pdf' % (fname_tag,
                                     configs.name))
    return trans_subset, top_syms


def query_syms_on_matrices(query_syms, trans_matrices, trans_names,
                           syms, configs, topn=8):
    top_syms_list = []
    for i, trans_mat in enumerate(trans_matrices):
        _, top_syms = query_syms_on_mat(query_syms, trans_mat, syms,
                                        configs, topn=topn)
        top_syms_list.extend(top_syms)
        print trans_names[i], top_syms
    top_syms_list = sorted(set(top_syms_list), key=top_syms_list.index)

    from subset_tools import subset
    matrices = None
    query_syms_list = []
    for query_sym in query_syms:
        for i, trans_mat in enumerate(trans_matrices):
            mat = subset(trans_mat, syms, [query_sym], top_syms_list)
            if matrices is None:
                matrices = mat
            else:
                matrices = np.vstack((matrices, mat))
            sym_name = '%s-%s' % (trans_names[i], query_sym)
            query_syms_list.append(sym_name)
    matrices = np.squeeze(matrices)
    print 'matrices', matrices.shape

    plot_mat(matrices, '', top_syms_list, query_syms_list)
    fname_tag = '_'.join(query_syms)
    fname_tag = fname_tag.replace('/', '_')
    plt.savefig('trans-both-%s-%s.pdf' % (fname_tag,
                                          configs.name))


def query_syms_on_trans():
    ngram_trans, nn_trans, ngram, nn, configs, data = get_trans()
    # trans = [nn_trans, ngram_trans]
    trans = [np.log(nn_trans), np.log(ngram_trans)]
    trans_names = ['nn', 'ngram']

    # query_syms = ['v', 'vii/o7']
    # query_syms = ['v']
    query_syms = ['vii/o7']  # , 'viio6']#, 'IV6']

    top_syms = []
    next_top_syms = []

    for sym in query_syms:
        print '-- %s ---' % sym
        topn = 10
        similars = nn.most_similar(sym, topn=topn)

        for i, sim in enumerate(similars):
            print sim
            if i > topn/2:
                top_syms.append(sim[0])
            else:
                next_top_syms.append(sim[0])

    nn.plot_w1(query_syms + top_syms)
    nn.plot_w1(query_syms + next_top_syms)
    # syms = data.syms
    # print 'ngram counts'
    # print ngram.n_gram_counts[syms.index(query_syms[0]), :]

    query_syms_on_matrices(query_syms + top_syms, trans, trans_names,
                           data.syms, configs)
    query_syms_on_mat(query_syms+top_syms, nn_trans, data.syms, configs)

    query_syms_on_matrices(query_syms + next_top_syms, trans, trans_names,
                           data.syms, configs)

    query_syms_on_mat(query_syms+next_top_syms, nn_trans, data.syms, configs)

if __name__ == '__main__':
    # predict_probs_seq()
    # compare_transition_matrices()
    # query_syms_on_trans()
    get_models()
