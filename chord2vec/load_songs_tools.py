
import os
import cPickle as pickle

import numpy as np

from config import *
from PreprocessedData import PreprocessedData, Seq


def load_songs(configs):
    # Load 173? chord sequences from Rolling Stone "500 Greatest Songs of All Time"
    print configs['corpus']

    if configs['corpus'] == 'bach':
        fname = os.path.join('data', 'bach_chorales_rn.pkl')
        if configs['use_durations']:
            fname = os.path.join('data', 'bach_chorales_rn_durs.pkl')
        if configs['augmented_data']:
            fname = os.path.join('data', 'bach-letters-augmented.pkl')

        # fname = os.path.join('data', 'bach_chorales_rn_seqintclass_durs.pkl')

        # with open(fname, 'rb') as p:
        #     seqs = pickle.load(p)
        #     durations = pickle.load(p)
        #     clipped_durs = []
        #     for durs in durations:
        #         durs = np.asarray(durs)
        #         durs[durs>2.0] = 2.0
        #         durs = list(durs)
        #         clipped_durs.append(durs)
        # return seqs, clipped_durs

        # fname = os.path.join('data', 'bach_chorales_rn_alone.pkl')

    elif configs['corpus'] == 'rock' and not configs['use_letternames']:
        fname = os.path.join('data', 'rock-rns.pkl')

    elif configs['corpus'] == 'rock' and configs['use_letternames'] and not configs['transposed']:
        fname = os.path.join('data', 'rock-lettername-originalKey.pkl')
    elif configs['corpus'] == 'rock' and configs["use_letternames"]:
        fname = os.path.join('data', 'rock_letternames_fixed.pkl')
        if configs['augmented_data']:
            fname = os.path.join('data', 'rock-augmented.pkl')
    else:
        assert False, 'ERROR: Data set configuration not available'

    print 'fname', fname
    path = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(path, fname)
    print fname
    durs = None
    with open(fname, 'rb') as p:
        seqs = pickle.load(p)
        if configs["use_durations"]:
            durs = pickle.load(p)
    print 'num of songs:', len(seqs)
    if configs["use_durations"]:
        assert durs is not None, 'ERROR: not yet supporting duration'
        # return seqs, durs
    else:
        return seqs


def get_segmented_songs(seqs=None, min_len=5):
    if seqs is None:
        from config import get_configs
        configs = get_configs()
        seqs = load_songs(configs)
    subseqs = []
    for seq in seqs:
        subseq = []
        for s in seq:
            subseq.append(s)
            if (s == 'I' or s == 'i') and len(subseq) > min_len:
                subseqs.append(subseq)
                subseq = []
    # with open('subseqs.txt', 'w') as p:
    #     lines = ''
    #     for seq in subseqs:
    #         line = ''
    #         for s in seq:
    #             line += '%s ' % s
    #         line += '\n'
    #         lines += line
    #     p.writelines(lines)
    return subseqs


def get_raw_data(configs=None):
    if configs is None:
        configs = get_configs()
    if configs['use_durations']:
        seqs, durs = load_songs(configs)
        sentences = [Seq(seqs[i], durs[i]) for i in range(len(seqs))]
    else:
        sentences = load_songs(configs)

    print sentences[0]

    from word2vec_utility_tools import build_vocab
    print "# of sentences", len(sentences)

    vocab2index, index2word = build_vocab(sentences,
                                          configs['min_count'])
    print "# of syms", len(index2word)
    print "syms:", index2word
    return sentences, index2word


def get_raw_encoded_data(configs=None):
    if configs is None:
        configs = get_configs()
    sentences, syms = get_raw_data(configs)
    seqs = []
    for sent in sentences:
        seq = [syms.index(s) for s in sent]
        seqs.append(seq)
    return seqs, syms


def pickle_train_test_seqs():
    configs = get_configs()
    data = get_data(configs)
    syms = data.syms
    train_seqs = data.get_train_seqs_data().seqs
    test_seqs = data.get_test_seqs_data().seqs
    print '# training:', len(train_seqs)
    print '# testing:', len(test_seqs)

    train_data = dict(seqs=train_seqs, syms=syms)
    fname = 'rock-train.pkl'
    path = os.path.join('data', 'chords', 'rock', 'train')
    if not os.path.isdir(path):
        os.makedirs(path)
    fpath = os.path.join(path, fname)
    print fpath
    with open(fpath, 'wb') as p:
        pickle.dump(train_data, p)
    save_as_text(fpath, train_seqs)

    test_data = dict(seqs=test_seqs, syms=syms)
    fname = 'rock-test.pkl'
    path = os.path.join('data', 'chords', 'rock', 'test')
    if not os.path.isdir(path):
        os.mkdir(path)
    fpath = os.path.join(path, fname)
    print fpath
    with open(fpath, 'wb') as p:
        pickle.dump(test_data, p)
    save_as_text(fpath, test_seqs)


def save_as_text(fpath, seqs):
    seqs_strs = []
    for seq in seqs:
        seq_str = ', '.join(seq) + '\n'
        seqs_strs.append(seq_str)
    fpath = os.path.splitext(fpath)[0] + '.txt'
    print 'save fname:', fpath
    with open(fpath, 'w') as p:
        p.writelines(seqs_strs)


def read_text(fpath):
    print fpath
    with open(fpath, 'r') as p:
        seqs = p.readlines()
    print '# of seqs:', len(seqs)
    return seqs


def read_seqs(fpath):
    lines = read_text(fpath)
    seqs = []
    for line in lines:
        if ', ' in line:
            syms = line.strip().split(', ')
        else:
            syms = line.strip().split(' ')
        assert ' ' not in syms
        seqs.append(syms)
    return seqs


def check_train_test_texts():
    pickle_train_test_seqs()

    path = os.path.join('data', 'chords', 'rock', 'train')
    train_fname = os.path.join(path, 'rock-train.txt')
    read_text(train_fname)

    path = os.path.join('data', 'chords', 'rock', 'test')
    test_fname = os.path.join(path, 'rock-test.txt')
    read_text(test_fname)


def get_train_test_data():
    configs = get_configs()
    data = get_data(configs)
    syms = data.syms
    print '# of symbols:', len(syms)
    train_seqs = data.get_train_seqs_data().seqs
    test_seqs = data.get_test_seqs_data().seqs
    print '# training:', len(train_seqs)
    print '# testing:', len(test_seqs)
    print train_seqs[0]
    return train_seqs, test_seqs, syms


def get_data(configs=None):
    if configs is None:
        configs = get_configs()
    seqs, syms = get_raw_data(configs)
    print 'get data, # of syms', len(syms), len(set(syms))
    assert len(syms) == len(set(syms))
    window = configs["window"]
    data = PreprocessedData(seqs, syms, window)
    return data


def get_configs_data():
    configs = get_configs()
    data = get_data(configs)
    return configs, data


def make_rn2letter_dict():
    data = get_data()
    syms = data.syms
    conversion_dict = {}
    from music21_chord_tools import roman2letter
    for sym in syms:
        # print sym
        conversion_dict[sym] = roman2letter(sym)

    fname = 'rn2letter.pkl'
    with open(fname, 'wb') as p:
        pickle.dump(conversion_dict, p)


def check_roman_vs_letters():
    data = get_data()
    syms = data.syms
    from music21_chord_tools import roman2letter, letter2roman
    mismatches = {}
    for sym in syms:
        letter = roman2letter(sym)
        roman = letter2roman(letter)
        if sym != roman:
            mismatches[roman] = sym
            print '------ mismatch'
    print mismatches


def test_get_raw_data():
    seqs, syms = get_raw_data()
    count_end_or_start_with_C = 0
    for seq in seqs:
        if seq[0] == 'C' or seq[-1] == 'C':
            count_end_or_start_with_C += 1
        else:
            print seq[0], seq[-1]
    print 'number of songs:', len(seqs)
    print 'count_end_or_start_with_C:', count_end_or_start_with_C


def convert_to_pickle(fname):
    fpath = os.path.join('data', fname)
    seqs = read_seqs(fpath)
    fname_parts = fname.split('.')
    assert len(fname_parts) == 2
    pickle_fname = fname.split('.')[0] + '.pkl'
    pickle_fpath = os.path.join('data', pickle_fname)
    with open(pickle_fpath, 'wb') as p:
        pickle.dump(seqs, p)


if __name__ == '__main__':
    # seqs = load_songs()
    # print len(seqs)
    # seqs = get_segmented_songs()
    # make_rn2letter_dict()
    # check_roman_vs_letters()

    # configs = get_configs()
    # seqs, syms = get_raw_data(configs)
    # print seqs[0]

    # pickle_train_test_seqs()
    # check_train_test_texts()
    # get_train_test_data()

    # test_get_raw_data()
    fname = 'rock-lettername-originalKey.txt'
    convert_to_pickle(fname)