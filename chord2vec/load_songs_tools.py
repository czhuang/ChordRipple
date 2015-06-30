
import os
import cPickle as pickle

from config import *
from PreprocessedData import PreprocessedData, Seq


def load_songs(configs):
    # Load 173? chord sequences from Rolling Stone "500 Greatest Songs of All Time"
    print configs['corpus']

    if configs['corpus'] == 'bach':
        fname = os.path.join('data', 'bach_chorales_rn.pkl')
        if configs['use_durations']:
            fname = os.path.join('data', 'bach_chorales_rn_durs.pkl')
        if configs['augment_data']:
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

    elif configs["use_letternames"] and not configs["use_original"]:
        fname = os.path.join('data', 'rock_chord_strs_notenames_basic.pkl')
    elif configs["use_letternames"]:
        fname = os.path.join('data', 'rock_letternames_fixed.pkl')
        if configs['augmented_data']:
            fname = os.path.join('data', 'rock-augmented.pkl')

    elif configs["use_original"]:
        fname = os.path.join('data', 'rock_chord_strs_original.pkl')

    else:
        fname = os.path.join('data', 'rock_chord_strs.pkl')

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


def get_raw_data(configs):
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
    print index2word
    return sentences, index2word


def get_data(configs=None):
    if configs is None:
        configs = get_configs()
    seqs, syms = get_raw_data(configs)
    print 'get data, # of syms', len(syms)
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


if __name__ == '__main__':
    # seqs = load_songs()
    # print len(seqs)
    # seqs = get_segmented_songs()
    # make_rn2letter_dict()
    check_roman_vs_letters()
