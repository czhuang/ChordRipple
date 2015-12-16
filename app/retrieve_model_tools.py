
import os

import cPickle as pickle

import numpy as np

from NGram import NGram
from retrieve_SkipGram_weights import retrieve_chord2vec_weights

from config import get_configs


def update_syms(configs, symbol_fname, syms):
    # retrieve symbol mapping dictionary
    fpath = os.path.join('data', symbol_fname)
    with open(fpath, 'rb') as p:
        rn2letter = pickle.load(p)
    letter2rn = {}
    for k, v in rn2letter.iteritems():
        letter2rn[v] = k

    # TODO: quick hack to check to see if the original syms were roman
    is_rn = True
    rn_count = 0
    rns = ['i', 'v', 'V', 'I']
    for sym in syms[:5]:
        is_rn_local = False
        for rn in rns:
            if rn in sym:
                is_rn_local = True
                rn_count += 1
                break

    if rn_count < 3:
        is_rn = False

    # convert symbol to requested symbol type
    if is_rn and configs['use_letternames']:
        syms = [rn2letter[sym] for sym in syms]
    elif not is_rn and not configs['use_letternames']:
        syms = [letter2rn[sym] for sym in syms]

    return rn2letter, letter2rn, syms


def retrieve_NGram(return_pickle=False):
    configs = get_configs()
    if configs['corpus'] != 'rock':
        fname = 'bigram-bach.pkl'
        symbol_fname = 'rn2letter-bach.pkl'
    else:
        fname = 'bigram-rock.pkl'
        symbol_fname = 'rn2letter-rock.pkl'
    print '...retrieve_NGram fname', fname
    ngram = NGram(fname=fname)

    ngram.rn2letter, ngram.letter2rn, ngram.syms = \
        update_syms(configs, symbol_fname, ngram.syms)

    print 'retrieve_NGram, # of syms', len(ngram.syms)

    return ngram


class SkipGramNNWrapper():

    def __init__(self, which_embedding='CG'):
        configs = get_configs()
        if configs['corpus'] != 'rock':
            fname = 'embedding-skipgram-bach.pkl'
            symbol_fname = 'rn2letter-bach.pkl'
        else:
            # fname = 'embedding-skipgram-rock.pkl'
            fname = 'embedding-rock-rn-10-500.pkl'
            symbol_fname = 'rn2letter-rock.pkl'

        if which_embedding == 'CG':
            print '...SkipGramNNWrapper, directory', os.getcwd()
            fpath = os.path.join('data', fname)
            print 'fname: ', fpath
            with open(fpath, 'rb') as p:
                embedding_dict = pickle.load(p)
            assert 'W1' in embedding_dict.keys() and \
                'syms' in embedding_dict.keys()
            for key, val in embedding_dict.iteritems():
                setattr(self, key, val)
        elif which_embedding == 'SGD':
            self.W1, self.syms = retrieve_chord2vec_weights()

        # normalize W1
        norm = np.linalg.norm(self.W1, axis=1)
        self.W1_norm = self.W1 / norm[:, None]

        self.rn2letter, self.letter2rn, self.syms = \
            update_syms(configs, symbol_fname, self.syms)

        print 'SkipGramNNWrapper, # of syms', len(self.syms)



    def get_vec(self, sym):
        if sym in self.syms:
            ind = self.syms.index(sym)
            return self.W1[ind, :]
        else:
            print 'WARNING: %s not in vocabulary' % sym

    def norm_vec(self, sym):
        if sym not in self.syms:
            return None
        ind = self.syms.index(sym)
        vec = self.W1[ind]
        norm = np.linalg.norm(vec)
        unit_vec = vec/norm
        return unit_vec

    def norm(self, sym):
        if sym not in self.syms:
            return None
        ind = self.syms.index(sym)
        vec = self.W1[ind]
        norm = np.linalg.norm(vec)
        return norm

    def theta(self, sym):
        if sym not in self.syms:
            return None
        ref_vec = np.ones((self.W1.shape[1]))
        unit_ref_vec = ref_vec / np.linalg.norm(ref_vec)
        unit_vec = self.norm_vec(sym)
        # print unit_vec
        # print unit_ref_vec
        cosine = np.dot(unit_vec, unit_ref_vec)
        theta = np.arccos(cosine)
        return theta

    def thetas(self, seqs):
        return [self.theta(sym) for sym in seqs]

    def most_similar(self, ref_sym, topn=3, return_scores=True):
        ref_unit_vec = self.norm_vec(ref_sym)
        if ref_unit_vec is None:
            return []
        dists = np.dot(self.W1_norm, ref_unit_vec)
        best = np.argsort(-dists)[:topn+1]
        ref_ind = self.syms.index(ref_sym)
        assert best[0] == ref_ind
        result = [(self.syms[ind], float(dists[ind])) for ind in best if ind != ref_ind]
        top_sims = result[:topn]
        if return_scores:
            return top_sims
        else:
            return [sim[0] for sim in top_sims]

    def most_diff(self, ref_sym, topn=3):
        ref_unit_vec = self.norm_vec(ref_sym)
        dists = np.dot(self.W1_norm, np.negative(ref_unit_vec))
        best = np.argsort(-dists)[:topn+1]
        ref_ind = self.syms.index(ref_sym)
        result = [(self.syms[ind], float(dists[ind])) for ind in best if ind != ref_ind]
        return result[:topn]


def retrieve_SkipGramNN_SGD():
    model = SkipGramNNWrapper('SGD')
    return model


def retrieve_SkipGramNN():
    model = SkipGramNNWrapper()
    return model


if __name__ == '__main__':
    ngram = retrieve_NGram()
    model = SkipGramNNWrapper()
    sim_syms = model.most_similar('I6/4')
    print sim_syms
