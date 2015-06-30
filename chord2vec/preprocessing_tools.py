

from music21_chord_tools import roman2letter_transpose, is_roman_numeral
import cPickle as pickle


def duplicate_by_transposing_and_into_letters(seqs):
    assert is_roman_numeral(seqs[0][0])
    print '# of seqs:', len(seqs)
    augmented_seqs = []
    translation_dict = {}

    for seq in seqs:
        for i in range(12):
            transposed_seq = []
            for sym in seq:
                letter = roman2letter_transpose(sym, -i,
                                                translation_dict=translation_dict)
                transposed_seq.append(letter)

            augmented_seqs.append(transposed_seq)
    print '# of augmented_seqs:', len(augmented_seqs)
    print 'translation_dict'
    for key, val in translation_dict.iteritems():
        print key, val
    for seq in augmented_seqs:
        print seq[:6]
    assert len(seqs)*12 == len(augmented_seqs)
    return augmented_seqs, translation_dict


def test_duplicate_by_transposing_and_into_letters():
    from config import get_configs
    from load_songs_tools import get_raw_data
    configs = get_configs()
    configs["augment_data"] = False
    configs["use_letternames"] = False
    seqs, syms = get_raw_data(configs)
    seqs, translation_dict = duplicate_by_transposing_and_into_letters(seqs)

    # want to store augmented_seqs
    fname_base = '%s-letters-augmented' % (configs['corpus'])

    print '...writing chord strings to file'
    fname = '%s.txt' % fname_base

    with open(fname, 'w') as p:
        for seq in seqs:
            for ch in seq:
                p.write(ch + ' ')
            p.write('\n')
    print '.finished writing chord strings to file'

    print '...pickling chord strings to file'
    fname = '%s.pkl' % fname_base
    with open(fname, 'wb') as p:
        pickle.dump(seqs, p)
    print '.finished pickling chord strings to file'

    print '... pickling translation dict to file'
    fname = '%s-translation_dict.pkl' % fname_base
    with open(fname, 'wb') as p:
        pickle.dump(translation_dict, p)
    print '.finished pickling translation dict to file'


def load_translation_dict():
    fname = 'bach-letters-augmented-translation_dict.pkl'
    with open(fname, 'r') as p:
        trans_dict = pickle.load(p)
    for key, val in trans_dict.iteritems():
        print key, val


if __name__ == '__main__':
    test_duplicate_by_transposing_and_into_letters()
    # load_translation_dict()
