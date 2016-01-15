
import os
import cPickle as pickle

from music21_chord_tools import *
from load_songs_tools import read_seqs, save_as_text


def get_original_rock_seqs():
    fname = os.path.join('data', 'rock-rns.pkl')
    path = os.path.dirname(os.path.realpath(__file__))
    fpath = os.path.join(path, fname)
    print fpath
    with open(fpath, 'rb') as p:
        seqs = pickle.load(p)
    return seqs


def get_original_bach_seqs():
    fname = os.path.join('data', 'bach_chorales_rn.pkl')
    path = os.path.dirname(os.path.realpath(__file__))
    fpath = os.path.join(path, fname)
    print fpath
    with open(fpath, 'rb') as p:
        seqs = pickle.load(p)
    return seqs


def collect_syms(seqs):
    syms = []
    for seq in seqs:
        for s in seq:
            if s not in syms:
                syms.append(s)
    return syms


def add_new_symbols_to_harmony():
    harmony.addNewChordSymbol('7sus4', '1,4,5,b7', ['7sus4'])


def convert_rock_syms_to_letter():
    seqs = get_original_rock_seqs()
    sym2letter = convert_syms_to_letter(seqs)
    fname = 'rn2letter-rock.pkl'
    fname = os.path.join('data', fname)
    with open(fname, 'wb') as p:
        pickle.dump(sym2letter, p)


def convert_bach_syms_to_letter():
    # TODO: breaks on 'bVII7[maj7]'
    seqs = get_original_bach_seqs()
    sym2letter = convert_syms_to_letter(seqs)
    fname = 'rn2letter-bach.pkl'
    fname = os.path.join('data', fname)
    with open(fname, 'wb') as p:
        pickle.dump(sym2letter, p)


def convert_syms_to_letter(seqs):
    syms = collect_syms(seqs)
    # add_new_symbols_to_harmony()
    print syms
    sym2letter = {}
    for sym in syms:
        print sym,
        letter = roman2letter(sym)
        print letter
        sym2letter[sym] = letter
    return sym2letter






def retrieve_rn2letter_dict():
    fname = 'rn2letter-rock.pkl'
    fname = os.path.join('data', fname)
    with open(fname, 'rb') as p:
        rn2letter = pickle.load(p)
    print rn2letter
    print '# of syms', len(rn2letter)
    return rn2letter


def encode_rock_to_letternames():
    seqs = get_original_rock_seqs()
    rn2letter = retrieve_rn2letter_dict()
    lettername_seqs = []
    for i, seq in enumerate(seqs):
        lettername_seq = [rn2letter[s] for s in seq]
        lettername_seqs.append(lettername_seq)

    # print lettername_seqs[:2]

    fname_tag = 'rock_letternames_fixed'
    fname = os.path.join('data', '%s.pkl' % fname_tag)
    path = os.path.dirname(os.path.realpath(__file__))
    fpath = os.path.join(path, fname)
    print fpath
    with open(fpath, 'wb') as p:
        pickle.dump(lettername_seqs, p)

    fname = os.path.join('data', 'rock_letternames_fixed.txt')
    seqs_strs = []
    for seq in seqs:
        seq_str = ', '.join(seq) + '\n'
        seqs_strs.append(seq_str)
    with open(fname, 'w') as p:
        p.writelines(seqs_strs)

    return seqs


def transpose_lettername(sym1, intval):
    ch1 = get_chordSymbol(sym1)
    ch1_transposed = ch1.transpose(intval)
    ch1_sym = harmony.chordSymbolFromChord(ch1_transposed)
    return ch1_sym.figure


def compute_transpose_interval(sym1, sym2):
    ch1 = get_chordSymbol(sym1)
    ch2 = get_chordSymbol(sym2)
    print ch1, ch2
    intval = interval.Interval(ch2.bass().midi - ch1.bass().midi)
    return intval


def encode_rock_original_lettername():
    path = 'data'
    # read in transposed to C chord sequences
    #fname = 'rock_letternames_fixed.txt'

    # read in the roman numerals
    fname = 'rock-rns.txt'
    fpath = os.path.join(path, fname)
    seqs = read_seqs(fpath)
    print seqs[0]

    # hack
    # read in chord sequences just to use the first chord to get transpose interval
    # for original key,
    # can't use as is because chord symbols are encoded in a different representation that's less readable
    # only need to read it's base
    fname = 'rock_chord_strs_notenames-edited.txt'
    fpath = os.path.join(path, fname)
    ref_seqs = read_seqs(fpath)
    print ref_seqs[0]

    transposed_back_seqs = []
    start_ind = 0
    # end_ind = start_ind + 1
    end_ind = 173

    seqs_slice = slice(start_ind, end_ind)
    # for i, seq in enumerate(seqs):
    for i, seq in zip(range(start_ind, end_ind), seqs[seqs_slice]):
        print 'seq', i, seq[0], ref_seqs[i][0]
        letter = roman2letter(seq[0])
        intval = compute_transpose_interval(letter, ref_seqs[i][0])
        transposed_back_seq = []
        for sym in seq:
            letter = roman2letter(sym)
            sym_transposed = transpose_lettername(letter, intval)
            transposed_back_seq.append(sym_transposed)
        transposed_back_seqs.append(transposed_back_seq)
        print transposed_back_seq[:10]
        print ref_seqs[i][:10]

    fname = 'rock-lettername-originalKey.txt'
    fpath = os.path.join(path, fname)
    save_as_text(fpath, transposed_back_seqs)

    # just for checking
    for i in range(10):
        print '----'
        print transposed_back_seqs[i][:10]
        print ref_seqs[i][:10]

    return transposed_back_seqs


def check_double_flats():
    path = 'data'
    # read in the roman numerals
    fname = 'rock-rns.txt'
    fpath = os.path.join(path, fname)
    seqs = read_seqs(fpath)
    print seqs[0]

    # read in letternames original key
    fname = 'rock-lettername-originalKey.txt'
    fpath = os.path.join(path, fname)
    seqs_transposed = read_seqs(fpath)

    double_flats_seq_inds = []
    for i, seq in enumerate(seqs_transposed):
        for sym in seq:
            if '--' in sym:
                double_flats_seq_inds.append(i)
                break
    print double_flats_seq_inds

    for i in double_flats_seq_inds:
        print '--'
        print seqs[i]
        print seqs_transposed[i]
        for j, sym in enumerate(seqs_transposed[i]):
            if '--' in sym:
                print '(', seqs[i][j], ',', seqs_transposed[i][j], ')'
        print


if __name__ == '__main__':
    # convert_rock_to_letternames()
    # convert_syms_to_letter()
    # roman2letter('v7s4')

    # run both lines if update dict
    # convert_syms_to_letter()
    # encode_rock_to_letternames()

    # check_user_system_conversion()
    # encode_rock_original_lettername()
    # check_double_flats()

    convert_bach_syms_to_letter()
