

from preprocessing_songs_tool import *


def check_roman2letter(sym):
    print '...', sym,
    try:
        letter = roman2letter_subroutine(sym)
    except:
        letter = None
    return letter


def check_convert_syms_to_letter():
    seqs = get_original_rock_seqs()
    syms = collect_syms(seqs)
    print syms
    problems = []
    for sym in syms:
        print sym,
        letter = check_roman2letter(sym)
        print letter
        if letter is None:
            problems.append(sym)
    print problems
    return problems


def check_user_system_conversion():
    from music21_chord_tools import preprocess_letters_before_sym2chord
    rn2letter = retrieve_rn2letter_dict()
    syms = rn2letter.values()
    for sym in syms:
        print sym,
        user_sym = sym.replace('-', 'b')
        music21_sym = preprocess_letters_before_sym2chord(user_sym)
        print user_sym, music21_sym
        assert sym == music21_sym


def check_pkl_seqs():
    fname = os.path.join('data', 'rock_letternames_fixed.pkl')
    with open(fname, 'rb') as p:
        seqs = pickle.load(p)

    fname = os.path.join('data', 'rock_letternames_fixed.txt')
    seqs_strs = []
    for seq in seqs:
        seq_str = ', '.join(seq) + '\n'
        print seq_str
        seqs_strs.append(seq_str)
    with open(fname, 'w') as p:
        p.writelines(seqs_strs)


def test_transpose():
    sym1 = 'C'
    sym2 = 'F'
    sym3 = 'Am7'
    intval = compute_transpose_interval(sym1, sym2)
    print 'intval', intval
    sym3_tranposed = transpose_lettername(sym3, intval)
    print sym1, sym2
    print sym3, sym3_tranposed


def test_check():
    sym, postfix = check('G7s4')
    print sym, postfix


if __name__ == '__main__':
    # check_pkl_seqs()
    # print roman2letter('v7s4')
    # check_user_system_conversion()
    # test_transpose()
    test_check()
