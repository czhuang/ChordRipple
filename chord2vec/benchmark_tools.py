
import os
import numpy as np

import kenlm

from collections import defaultdict

from load_model_tools import get_models
from config import get_configs
from load_songs_tools import get_data

from NGram import NGram


def get_train_test_seq_syms():
    configs = get_configs()
    data = get_data(configs)
    syms = data.syms
    train_seqs = data.get_train_seqs_data().seqs
    test_seqs = data.get_test_seqs_data().seqs

    assert data.get_train_seqs_data().syms == data.syms
    print '# training:', len(train_seqs)
    print '# testing:', len(test_seqs)

    return train_seqs, test_seqs, syms


def ngram_performance():
    configs = get_configs()
    train_seqs, test_seqs, syms = get_train_test_seq_syms()
    ngram = NGram(train_seqs, syms,
                  configs=configs)
    ngram.score(test_seqs)


def ngram_generate_seqs(train_seqs, syms, num_seq=20, seq_length=15):
    configs = get_configs(print_config=False)
    # train_seqs, test_seqs, syms = get_train_test_seq_syms()
    ngram = NGram(train_seqs, syms,
                  configs=configs)
    print 'Generating Chords from bi-gram model...'
    for i in range(num_seq):
        seq = ngram.gen_seq(seq_length)
        print ' '.join(seq)


def smoothed_ngram_performance():
    ns = range(2, 5)
    train_seqs, test_seqs, syms = get_train_test_seq_syms()
    path = os.path.join('models', 'rock-letter')
    scores = defaultdict(list)
    for n in ns:
        fname = 'rock-%d.arpa' % n
        fpath = os.path.join(path, fname)
        print 'fpath:', fpath
        model = kenlm.LanguageModel(fpath)
        order = model.order
        assert order == n
        for seq in test_seqs:
            seq_str = ' '.join(seq)
            score = model.score(seq_str)
            # collaspsing scores of all seq into one seq
            scores[n].append(score)

    seq_lens = [len(seq) for seq in test_seqs]
    seq_len = np.sum(seq_lens)
    print '# of seq:', len(test_seqs)
    print 'seq_len:', seq_len

    for n in ns:
        print '%d-gram score: %.2f' % (n, np.sum(scores[n]))
        print 'average %d-gram score: %.4f' % (n, np.sum(scores[n])/seq_len)


if __name__ == '__main__':
    ngram_performance()
    # smoothed_ngram_performance()
