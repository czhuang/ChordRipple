

from music21 import *
import numpy as np
from music21_chord_tools import root_interval
from load_songs_tools import get_train_test_data

END_SYMBOL = ''


# def add_attributes(seqs):
#     for seq in seqs:
#         # attributes
#         attributes = np.zeros(())


def compute_interval(sym1, sym2):
    interval_set = range(-5, 7)
    if sym1 == END_SYMBOL or sym2 == END_SYMBOL:
        semitones = 0
    else:
        ch1 = harmony.ChordSymbol(sym1)
        ch2 = harmony.ChordSymbol(sym2)
        intval = interval.notesToChromatic(ch1.bass(), ch2.bass())
        semitones = intval.semitones
        if np.abs(semitones) > 6:
            if semitones < 0:
                semitones += 12
            else:
                semitones -= 12
    return semitones


def compute_seq_interval(seq):
    intervals = [compute_interval(sym1, sym2)
                 for sym1, sym2 in zip(seq, seq[1:])]
    # the first being null, encoded also as 0 here
    # seq interval attribute is the interval from the chord before
    intervals = ([0] + intervals + [0] * len(seq))[:len(seq)]
    return np.asarray(intervals)


def test_interval_with_data():
    train_seqs, test_seqs, syms = get_train_test_data()
    for seq in train_seqs:
        for sym1, sym2 in zip(seq, seq[1:]):
            print sym1, sym2
            if sym2 != END_SYMBOL:
                print compute_interval(sym1, sym2)


def check_syms():
    train_seqs, test_seqs, syms = get_train_test_data()
    for sym in syms:
        if sym == END_SYMBOL:
            continue
        print harmony.ChordSymbol(sym)


def test_interval():
    syms = ['C', 'D', 'E', 'F#', 'F', 'G', 'A', 'B', 'C']
    for sym1, sym2 in zip(syms, syms[1:]):
        print sym1, sym2, compute_interval(sym1, sym2)

    for sym2 in syms:
        print syms[0], sym2, compute_interval(sym1, sym2)

    for sym2 in syms:
        print syms[-2], sym2, compute_interval(sym1, sym2)

if __name__ == "__main__":
    test_interval()
    # test_interval_with_data()
    # check_syms()