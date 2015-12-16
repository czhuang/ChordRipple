
import cPickle as pickle

import numpy as np

from attribute_tools import compute_seq_interval


class Data(object):
    def __init__(self, seq_length, num_seq, retrieve=False):
        self.seq_length = seq_length
        self.num_seq = num_seq

        from load_songs_tools import get_train_test_data
        self.train_seqs, self.test_seqs, self.syms = get_train_test_data()

        print('sym size:', self.vocab_sz)
        self.train_inputs = self.build_data_repr(self.train_seqs, "train", retrieve)
        self.test_inputs = self.build_data_repr(self.test_seqs, "test", retrieve)
        self.input_size = self.train_inputs.shape[2]
        self.output_size = len(self.syms)
        self.num_attributes = self.input_size - self.vocab_sz

    @property
    def vocab_sz(self):
        return len(self.syms)

    def string_to_one_hot(self, seq):
        # TODO: assumes all symbols are seen
        inds = np.array([self.syms.index(s) for s in seq]).T
        return np.array(inds[:, None] == np.arange(self.vocab_sz)[None, :], dtype=int)

    def string_to_full_vec(self, seq):
        one_hots = self.string_to_one_hot(seq)
        attributes = self.compute_attributes(seq)
        np.h

    def one_hot_to_string(self, one_hot_matrix):
        ch_encoding = self.get_targets(one_hot_matrix)
        return [self.syms[np.argmax(c)] for c in ch_encoding]

    def get_targets(self, data):
        if self.vocab_sz == data.shape[-1]:
            return data
        if len(data.shape) == 2:
            return data[:, :-self.num_attributes]
        targets = data[:, :, :-self.num_attributes]
        assert self.vocab_sz == targets.shape[-1]
        return targets

    def compute_attributes(self, seq):
        return compute_seq_interval(seq)

    def build_data_repr(self, seqs, which_set, retrieve):
        num_attributes = 1
        fname = '%s_data_numAttr-%d.pkl' % (which_set, num_attributes)
        seqs_subset = seqs[:self.num_seq]
        # shape (time, which_seq, which_chord)
        data_shape = (self.seq_length, len(seqs_subset),
                      len(self.syms) + num_attributes)
        print 'data_shape:', data_shape
        if retrieve:
            print '...retrieving %s' % fname
            with open(fname, 'rb') as p:
                data = pickle.load(p)
            assert data.shape == data_shape
            return data
        else:
            data = np.zeros(data_shape)
            for idx, seq in enumerate(seqs_subset):
                padded_seq = (seq + [''] * self.seq_length)[:self.seq_length]
                attr = compute_seq_interval(padded_seq)
                assert len(padded_seq) == len(attr)
                chord_encoddings = self.string_to_one_hot(padded_seq)
                data_vecs = np.hstack((chord_encoddings,
                                       attr[:, np.newaxis]))
                data[:, idx, :] = data_vecs
            print 'Data (shape)', data.shape

            print '...saving %s' % fname
            with open(fname, 'wb') as p:
                pickle.dump(data, p)
        return data
