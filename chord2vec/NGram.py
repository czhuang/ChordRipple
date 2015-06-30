
import os

from copy import copy

import cPickle as pickle

import numpy as np

from music21_chord_tools import sym2chord


class NGram(object):
    def __init__(self, seqs=None, syms=None, n=2,
                 configs=None, fname=None):
        # TODO: generalize to any N
        assert n == 2

        # TODO: fix hacky load vs init, caused by not including original seqs
        # TODO: some variables not inited or used
        if fname is not None:
            self.load(fname)
        else:
            self.seqs = seqs

            self.n = n
            self.syms = syms
            self.configs = configs

            from PreprocessedData import RawData
            self.data = RawData(seqs, syms)
            self.ngram, self.ngram_counts = self.make_ngram(seqs, syms, n)
            self.start_probs = self.make_start_probs(seqs, syms)
            self.unigram, self.unigram_counts = self.make_ngram(seqs, syms, 1)

    @property
    def trans(self):
        return self.ngram

    @property
    def starts(self):
        return self.start_probs

    @property
    def priors(self):
        return self.unigram

    def save(self, fname):
        # only save the summaries, not the original sequences
        save_dict = {'syms': self.syms,
                     'configs': self.configs,
                     'ngram': self.ngram,
                     'ngram_counts': self.ngram_counts,
                     'unigram': self.unigram,
                     'unigram_counts': self.unigram_counts,
                     'start_probs': self.start_probs}
        with open(fname, 'wb') as p:
            pickle.dump(save_dict, p)

    def load(self, fname):
        fpath = os.path.join('data', fname)
        print 'Ngram fpath', fpath
        with open(fpath, 'rb') as p:
            save_dict = pickle.load(p)
        for key, val in save_dict.iteritems():
            setattr(self, key, val)

    @staticmethod
    def make_ngram(seqs, syms, n):
        assert n == 2 or n == 1
        ngram = None
        ngram_counts = None
        if n == 2:
            ngram, ngram_counts = bigram(seqs, syms)
        elif n == 1:
            ngram, ngram_counts = unigram(seqs, syms)
        else:
            print 'WARNING: %s-gram not yet implemented' % n
        return ngram, ngram_counts

    @staticmethod
    def make_start_probs(seqs, syms):
        counts = np.zeros((len(syms),))
        # TODO: fix hacky smoothing
        counts += 0.01
        for seq in seqs:
            if seq[0] in syms:
                ind = syms.index(seq[0])
                counts[ind] += 1
        counts /= np.sum(counts)
        return counts

    @staticmethod
    def score(self, seqs, top_n=1):
        scores = []
        posts = []
        for seq in seqs:
            local_score = np.zeros((len(seq)))
            local_posts = np.zeros((len(seq)))
            if seq[0] in self.syms:
                ind = self.syms.index(seq[0])
                predicted_ind = np.argmax(self.start_probs)
                local_score[0] = predicted_ind == ind
                local_posts[0] = self.start_probs[predicted_ind]
            else:
                local_score[0] = 0.0
                local_posts[0] = 0.0
            for i in range(1, len(seq)):
                if seq[i-1] not in self.syms or seq[i] not in self.syms:
                    continue
                ind_previous = self.syms.index(seq[i-1])
                ind = self.syms.index(seq[i])
                predicted_inds = np.argsort(-self.ngram[ind_previous, :])[:top_n]
                local_score[i] = ind in predicted_inds
                local_posts[i] = self.ngram[ind_previous, ind]
            scores.append(local_score)
            posts.append(local_posts)
        print '\n=== NGram performance ==='
        print '=== top %d' % top_n
        mean_scores = [np.mean(score) for score in scores]
        mean_score_total = np.mean(mean_scores)
        print 'scores: %.3f' % mean_score_total

        mean_scores = [np.mean(score[2:]) for score in scores]
        mean_score = np.mean(mean_scores)
        print 'scores (without first, 2nd): %.3f' % mean_score

        first_scores = [score[0] for score in scores]
        print 'first score: %.3f' % np.mean(first_scores)

        second_scores = [score[1] for score in scores]
        print 'second score: %.3f' % np.mean(second_scores)

        posts_total = np.sum([np.sum(post) for post in posts])
        print 'likelihood (posterior probs): %.3f' % posts_total
        return posts_total, mean_score_total

    @staticmethod
    def predict(self, seq, log=True, return_all=False):
        ind = self.syms.index(seq[0])
        probs = [self.start_probs[ind]]
        probs_all = [self.start_probs]
        for i in range(1, len(seq)):
            if seq[i-1] in self.syms and seq[i] in self.syms:
                ind = self.syms.index(seq[i-1])

                ind_next = self.syms.index(seq[i])
                probs.append(self.ngram[ind, ind_next])

                probs_all.append(self.ngram[ind, :])
        if log:
            probs = np.log(probs)
        if return_all:
            return probs, probs_all
        else:
            return probs

    def sample_multinomial(self, probs):
        draws = np.random.multinomial(1, probs)
        next_ind = np.argmax(draws)
        return self.syms[next_ind]

    def sample_next(self, seq, m=1, return_chord=False):
        # m here is for length of context
        assert m == 1
        seq_len = len(seq)
        if seq_len == 0:
            ind = self.sample_multinomial(self.start_probs)
        elif seq_len < m:
            assert False, 'ERROR: Not yet implemented'
        else:
            ind = self.syms.index(seq[-m])
        sym = self.sample_multinomial(self.ngram[ind, :])
        if not return_chord:
            return sym
        else:
            return sym2chord(sym)

    def sample_start(self, return_chord=False):
        sym = self.sample_multinomial(self.start_probs)
        if not return_chord:
            return sym
        else:
            return sym2chord(sym)

    def gen_seq(self, n, return_chord=False):
        # returns chords not sym labels
        seq = [self.sample_start()]
        for i in range(1, n):
            seq.append(self.sample_next(seq))
        if not return_chord:
            return seq
        else:
            return [sym2chord(sym) for sym in seq]

    def gen_continuation(self, seq, n,
                         m=1, return_chord=False):
        seq_cont = copy(seq)
        for i in range(n):
            seq_cont.append(self.sample_next(seq_cont, m=m))
        if not return_chord:
            return seq_cont
        else:
            return [sym2chord(sym) for sym in seq_cont]


def unigram(seqs, syms):
    unigram_counts = np.zeros((len(syms)))
    # TODO: fix hacky smoothing
    unigram_counts += 0.01
    for seq in seqs:
        for i in range(len(seq)):
            sym = seq[i]
            if sym in syms:
                ind = syms.index(sym)
                count = 1
                if hasattr(seq, 'weights'):
                    count = seq.weights[i]
                unigram_counts[ind] += count
    unigram_probs = unigram_counts / np.sum(unigram_counts)
    return unigram_probs, unigram_counts


def bigram(seqs, syms, row_syms=None, forward=True):
    if row_syms is None:
        row_syms = syms
    bigram_counts = np.zeros((len(row_syms), len(syms)))
    # TODO: fix hacky smoothing
    bigram_counts += 0.01
    for seq in seqs:
        p_sym = seq[0]
        for i in range(1, len(seq)):
            sym = seq[i]
            if sym in row_syms and p_sym in row_syms:
                c_ind = syms.index(sym)
                if p_sym in row_syms:
                    row_ind = row_syms.index(p_sym)
                    count = 1
                    if forward:
                        if hasattr(seq, 'weights'):
                            count = seq.weights[i]
                        bigram_counts[row_ind, c_ind] += count
                    else:
                        if hasattr(seq, 'weights'):
                            count = seq.weights[i-1]
                        bigram_counts[c_ind, row_ind] += count
            p_sym = sym
    norm = np.sum(bigram_counts, axis=1)
    bigram_probs = bigram_counts / norm[:, None]
    assert np.allclose(np.sum(bigram_probs, axis=1), np.ones((len(row_syms))))
    return bigram_probs, bigram_counts
