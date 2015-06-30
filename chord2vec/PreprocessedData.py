
from copy import copy

import numpy as np


from word2vec_utility_tools import collect_valid_skip_grams, get_train_test_encodings


class RawData(object):
    def __init__(self, seqs, syms):
        self.seqs = seqs
        self.syms = syms


class PreprocessedData(object):
    def __init__(self, seqs, syms, window, seq_processed=None):
        self.seqs = seqs
        from load_songs_tools import get_segmented_songs
        self.subseqs = get_segmented_songs(seqs)
        self.syms = syms
        self.window = window
        if seq_processed is not None:
            self.seqs_processed = seq_processed
        else:
            self.seqs_processed = [PreprocessedSeq(seq, syms, window) for seq in seqs]
            self.ins, self.outs \
                = self.concatenate_data(self.seqs_processed)

            self.inputs, self.outputs, self.weights,\
                self.test_inputs, self.test_outputs, self.test_weights =\
                get_train_test_encodings(self.ins, self.outs, syms)

            self.keys = self.check_keys()
            self.sorted_keys = self.sort_keys()

    def find_subseq(self, subseq):
        seqs = []
        for seq in self.seqs:
            for i in range(len(seq)):
                all_match = True
                for j in range(len(subseq)):
                    if i+j >= len(seq) or subseq[j] != seq[i+j]:
                        all_match = False
                        break
                if all_match:
                    seqs.append(copy(seq))
                    break
        return seqs

    def show_subseq_context(self, subseq, context_len=1):
        seqs = []
        contexts = []
        seq_inds = []
        for seq_ind, seq in enumerate(self.seqs):
            seq_added = False
            for i in range(len(seq)):
                all_match = True
                for j in range(len(subseq)):
                    if i+j >= len(seq) or (subseq[j] != seq[i+j] and
                                           subseq[j] != '*'):
                        all_match = False
                        break
                if all_match:
                    if not seq_added:
                        seqs.append(copy(seq))
                        seq_inds.append(seq_ind)
                        seq_added = True
                    start_ind = i - context_len
                    end_ind = i + len(subseq) + context_len
                    contexts.append(seq[start_ind:end_ind])

        return seqs, contexts, seq_inds

    def get_train_seqs_data(self, proportion=0.8):
        # TODO: this test data does not align with the test skipgrams
        endi = int(len(self.seqs_processed)*proportion)
        train_seqs = self.seqs_processed[:endi]
        return PreprocessedData(self.seqs[:endi], self.syms, self.window,
                                train_seqs)

    def get_test_seqs_data(self, proportion=0.8):
        # TODO: this test data does not align with the test skipgrams
        endi = int(len(self.seqs_processed)*proportion)
        test_seqs = self.seqs_processed[endi:]
        return PreprocessedData(self.seqs[endi:], self.syms, self.window,
                                test_seqs)

    def get_test_data(self):
        return Data(self.test_inputs, self.test_outputs, self.syms)

    def check_keys(self):
        if isinstance(self.inputs, dict):
            assert self.inputs.keys() == self.outputs.keys()
        keys = copy(self.outputs.keys())
        assert self.outputs.values()[0].shape[1] == len(self.syms)
        return keys

    def sort_keys(self):
        ints = [int(key) for key in self.keys]
        sorted_ints = sorted(ints)
        return [str(key) for key in sorted_ints]

    @staticmethod
    def get_train_test_encodings(inputs, outputs, syms, proportion=0.8):
        return get_train_test_encodings(inputs, outputs, syms,
                                        proportion=proportion)

    @staticmethod
    def concatenate_data(seqs_processed):
        inputs = {}
        for seq in seqs_processed:
            for key in seq.ins.keys():
                inputs[key] = inputs.get(key, []) + seq.ins[key]
        outputs = {}
        for seq in seqs_processed:
            for key in seq.outs.keys():
                outputs[key] = outputs.get(key, []) + seq.outs[key]
        for key in inputs.keys():
            assert len(inputs[key]) == len(outputs[key])
        return inputs, outputs


class SymbolInt(int):
    def __new__(cls, val, weight):
        obj = super(SymbolInt, cls).__new__(cls, val)
        obj.weight = weight
        return obj


class Seq(list):
    def __init__(self, seq, weights):
        list.__init__(self, seq)
        # self.extend(seq)
        self.weights = weights


class PreprocessedSeq(object):
    def __init__(self, seq, syms, window):
        self.seq = seq
        if not isinstance(seq, Seq):
            weights = np.ones((len(seq)))
            self.seq = Seq(seq, weights)
        self.syms = syms
        self.window = window
        self.inputs, self.outputs, self.valid_offsets =\
            self.build_skip_gram(seq, syms, window)
        self.ins, self.outs =\
            self.collect_valid_skip_grams(self.inputs, self.outputs)

    def collect_valid_skip_grams(self, inputs, outputs):
        return collect_valid_skip_grams(inputs, outputs, self.seq.weights)

    @staticmethod
    def build_skip_gram(seq, syms, window):
        valid_offsets = []
        weights = []
        outputs = {}
        # test_outputs = {}
        # n_skipgram = window*2
        for pos, word in enumerate(seq):
            if word not in syms:
                valid_offsets.append([])
                # still need to add None to outputs
                for offset in range(-window, window+1):
                    if offset == 0:
                        continue
                    key = str(offset)
                    outputs[key] = outputs.get(key, []) + [None]
                continue
            reduced_window = np.random.randint(1, window+1)
            # if (input) word is in vocabulary
            local_valid_offsets = []
            for offset in range(-window, window+1):
                key = str(offset)
                pos2 = pos + offset

                # invalid_training_only = False
                if offset == 0:
                    continue
                elif pos2 < 0 or pos2 >= len(seq):
                    output = None
                elif offset < -reduced_window or offset > reduced_window:
                    output = None
                    # TODO: mark as invalid only for training,
                    # and need to be able to include this skipgram during testing
                    # invalid_training_only = True
                elif seq[pos2] not in syms:
                    output = None
                else:
                    output = syms.index(seq[pos2])

                if output is not None:  # or invalid_training_only:
                    local_valid_offsets.append(offset)
                outputs[key] = outputs.get(key, []) + [output]
            valid_offsets.append(local_valid_offsets)
        inputs = []
        for word in seq:
            if word in syms:
                inputs.append(syms.index(word))
            else:
                inputs.append(None)
        # check
        for key in outputs.keys():
            assert len(inputs) == len(outputs[key])
        return inputs, outputs, valid_offsets


class Data(object):
    def __init__(self, ins, outs, syms):
        self.inputs = ins
        self.outputs = outs
        self.syms = syms
        self.keys = self.check_keys()
        self.sorted_keys = self.sort_keys()

    def check_keys(self):
        if isinstance(self.inputs, dict):
            assert self.inputs.keys() == self.outputs.keys()
        keys = copy(self.outputs.keys())
        assert self.outputs.values()[0].shape[1] == len(self.syms)
        return keys

    def sort_keys(self):
        ints = [int(key) for key in self.keys]
        sorted_ints = sorted(ints)
        return [str(key) for key in sorted_ints]
