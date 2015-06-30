
import os
import cPickle as pickle

from collections import OrderedDict

from music21 import *


def get_original_rock_seqs():
    fname = os.path.join('data', 'rock-rns.pkl')
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


def convert_syms_to_letter():
    seqs = get_original_rock_seqs()
    syms = collect_syms(seqs)
    # add_new_symbols_to_harmony()
    print syms
    sym2letter = {}
    for sym in syms:
        print sym,
        letter = roman2letter(sym)
        print letter
        sym2letter[sym] = letter
    fname = 'rn2letter-rock.pkl'
    fname = os.path.join('data', fname)
    with open(fname, 'wb') as p:
        pickle.dump(sym2letter, p)


def retrieve_rn2letter_dict():
    fname = 'rn2letter-rock.pkl'
    fname = os.path.join('data', fname)
    with open(fname, 'rb') as p:
        rn2letter = pickle.load(p)
    print rn2letter
    print '# of syms', len(rn2letter)
    return rn2letter


def roman2letter_subroutine(sym):
    rn = roman.RomanNumeral(sym)
    # print rn
    ch = chord.Chord(rn.pitches)
    # print ch
    cs = harmony.chordSymbolFromChord(ch)
    # print cs
    # print cs.figure
    return cs.figure


def check(sym):
    fixes = OrderedDict()
    fixes['7s4'] = 'sus4'
    fixes['s4'] = 'sus4'
    print fixes.keys()
    postfix = None
    sym = sym.replace('x', 'o')
    partial_sym = sym
    for k, v in fixes.iteritems():
        if k in sym:
            ind = sym.index(k)
            print 'ind', ind
            partial_sym = sym[:ind]
            partial_sym = partial_sym.upper()
            postfix = v
            break
    return partial_sym, postfix


def roman2letter(sym):
    print '...', sym,
    partial_sym, postfix = check(sym)
    print 'partial_sym', partial_sym, postfix
    letter = roman2letter_subroutine(partial_sym)
    if postfix is not None:
        letter = letter + postfix
    print letter
    return letter


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


if __name__ == '__main__':
    # convert_rock_to_letternames()
    # convert_syms_to_letter()
    # roman2letter('v7s4')

    # run both lines if update dict
    # convert_syms_to_letter()
    # encode_rock_to_letternames()

    # check_user_system_conversion()
    pass
