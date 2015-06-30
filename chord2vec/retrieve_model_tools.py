
import os

import cPickle as pickle

import numpy as np

from NGram import NGram

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
                break
        rn_count += 1
    if rn_count < 3:
        is_rn = False

    # convert symbol to requested symbol type
    if is_rn and configs['use_letternames']:
        syms = [rn2letter[sym] for sym in syms]
    elif not is_rn and not configs['use_letternames']:
        syms = [letter2rn[sym] for sym in syms]

    return rn2letter, letter2rn, syms


def retrieve_NGram():
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

    def __init__(self):
        configs = get_configs()
        if configs['corpus'] != 'rock':
            fname = 'embedding-skipgram-bach.pkl'
            symbol_fname = 'rn2letter-bach.pkl'
        else:
            # fname = 'embedding-skipgram-rock.pkl'
            fname = 'embedding-rock-rn-10-500.pkl'
            symbol_fname = 'rn2letter-rock.pkl'

        print '...SkipGramNNWrapper, directory', os.getcwd()
        fpath = os.path.join('data', fname)
        print 'fname: ', fpath
        with open(fpath, 'rb') as p:
            embedding_dict = pickle.load(p)
        assert 'W1' in embedding_dict.keys() and \
            'syms' in embedding_dict.keys()
        for key, val in embedding_dict.iteritems():
            setattr(self, key, val)
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

    def most_similar(self, ref_sym, topn=3):
        if ref_sym not in self.syms:
            return None
        ref_ind = self.syms.index(ref_sym)
        ref_vec = self.W1[ref_ind]
        ref_norm = np.linalg.norm(ref_vec)
        ref_unit_vec = ref_vec/ref_norm
        dists = np.dot(self.W1_norm, ref_unit_vec)
        best = np.argsort(-dists)[:topn+1]
        assert best[0] == ref_ind
        result = [(self.syms[ind], float(dists[ind])) for ind in best if ind != ref_ind]
        return result[:topn]


def retrieve_SkipGramNN():
    model = SkipGramNNWrapper()
    return model


if __name__ == '__main__':
    ngram = retrieve_NGram()
    model = SkipGramNNWrapper()
    sim_syms = model.most_similar('I6/4')
    print sim_syms
