
import os

import numpy as np

from NGram import NGram

import cPickle as pickle


def retrieve_NGram():
    fname = 'bigram-bach.pkl'
    ngram = NGram(fname=fname)
    return ngram


class SkipGramNNWrapper():
    def __init__(self):
        fname = 'embedding-skipgram-bach.pkl'
        print '...SkipGramNNWrapper, directory', os.getcwd()
        with open(fname, 'rb') as p:
            embedding_dict = pickle.load(p)
        assert 'W1' in embedding_dict.keys() and \
            'syms' in embedding_dict.keys()
        for key, val in embedding_dict.iteritems():
            setattr(self, key, val)
        # normalize W1
        norm = np.linalg.norm(self.W1, axis=1)
        self.W1_norm = self.W1 / norm[:, None]

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