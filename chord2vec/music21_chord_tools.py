
from music21 import roman, stream, chord, harmony, key, interval


letter_ordering = ['c', 'db', 'd', 'eb', 'e', 'f', 'gb', 'g', 'ab', 'a', 'bb', 'b']

# from resource.py
LETTER_PARTIAL_CO_RELABELS = [[['C##', 'D'], ['A#', 'Bb']],
                              [['C##', 'D'], ['E#', 'F']],
                              [['F##', 'G'], ['D#', 'Eb']],
                              [['F##', 'G'], ['D##', 'E']],
                              [['F##', 'G'], ['A#', 'Bb']],
                              [['G##', 'A'], ['B#', 'C']],
                              [['G##', 'A'], ['E#', 'F']],
                              [['G#', 'Ab'], ['B#', 'C']],
                              [['E#', 'F'], ['G#', 'Ab']],
                              [['E#', 'F'], ['D#', 'Eb']]]
LETTER_PARTIAL_RELABELS_FOR_USER = {'-': 'b',
                                    'C##': 'D',
                                    'E#': 'F',
                                    'B#': 'C'}

ROMAN2LETTER_RELABELS = {'bVII7[maj7]': 'B-maj7'}
LETTER2ROMAN_RELABELS = {'B-maj7': 'bVII7[maj7]'}
ROMAN_PARTIAL_RELABELS = {'6/5': '65', '4/3': '43', '6/4': '64'}
# LETTER_PARTIAL_RELABELS_FOR_USER = {'-': 'b'}
MAKE_NOTE_EXCEPTIONS = ['Vs4', 'V7s4']


# TODO: It6 treated as bVI7 as a quick hack
ROMAN_RELABELS = {'bVII7[maj7]': 'bVII7',
                  'It6': 'bVI7'}


def root_interval(sym, sym_next):
    # print sym, sym_next
    ch = harmony.ChordSymbol(sym)
    ch_next = harmony.ChordSymbol(sym_next)
    ch_root = ch.findRoot()
    ch_next_root = ch_next.findRoot()
    intvl = interval.Interval(ch_root, ch_next_root)
    semitones = intvl.chromatic.semitones
    return semitones

def is_fifth_root_movement_from_sym_pair(sym, sym_next):
    semitones = root_interval(sym, sym_next)
    # print sym, sym_next, semitones
    if semitones == -7 or semitones == 5:  # a fifth down
        return True
    return False

def is_fifth_root_movement(root, root_next):
    intvl = interval.Interval(root, root_next)
    semitones = intvl.chromatic.semitones
    if semitones == -7 or semitones == 5:  # a fifth down
        return True
    return False

def get_roots_and_chord_qualities(seq):
    roots = []
    chord_qualities = []
    for sym in seq:
        ch = harmony.ChordSymbol(sym)
        roots.append(ch.findRoot())
        if ch.isMajorTriad():
            chord_qualities.append('M')
        elif ch.isDominantSeventh():
            chord_qualities.append('d')
        else:
            chord_qualities.append('')
    return roots, chord_qualities


def get_targets(seq):
    # TODO: could use the quality of the chord to constrain this more
    # the last chord in a subsequence that's a fifth movement is a targe
    roots, chord_qualities = get_roots_and_chord_qualities(seq)
    fifth_mvts_booleans = []
    for i in range(len(seq)-1):
        fifth = is_fifth_root_movement(roots[i], roots[i+1])
        fifth_mvts_booleans.append(fifth)
    targets = {}
    if len(fifth_mvts_booleans) == 1:
        targets[seq[-1]] = 1
        # print targets
        return targets
    for i in range(len(fifth_mvts_booleans)):
        # print seq[i], seq[i+1],
        next_is_fifth = i < len(fifth_mvts_booleans) - 1 and \
                        fifth_mvts_booleans[i+1]
        last_fifth_mvt = fifth_mvts_booleans[i] and not next_is_fifth
        could_be_V = chord_qualities[i] == 'M' or chord_qualities[i] == 'd'
        # print last_fifth_mvt, could_be_V
        if last_fifth_mvt and could_be_V:
            target = seq[i+1]
            # print target, True
            # weighted by how many steps away from start
            if target not in targets:
                targets[target] = 1.0 / (i + 1)
            else:
                targets[target] += 1.0 / (i + 1)
        else:
            # just for printing debugging
            target = seq[i+1]
            # print target, False
    # print targets
    return targets

def letter2music21(sym):
    return replace_flat_dash_primary(sym, 0)


def is_roman_numeral(sym):
    signs_of_roman = ['i', 'I', 'v', 'V', 'It']
    signs_of_not_roman = ['dim']
    is_roman = False
    for sign in signs_of_roman:
        if sign in sym:
            is_roman = True
    for sign in signs_of_not_roman:
        if sign in sym:
            is_roman = False
    return is_roman


def sym2roman(sym):
    try:
        chord_sym = roman.RomanNumeral(sym)
        # print chord_sym
    except:
        print "WARNING: chord symbol not valid", sym
        chord_sym = None
    return chord_sym


def replace_flat_dash_primary(sym, part_ind):
    if len(sym) > 1 and sym[1] == 'b' and sym[0] != 'b':
        sym = sym.replace('b', '-')
    elif 'bb' == sym[:2]:
        if len(sym) > 2:
            sym = sym[0] + '-' + sym[2:]
        else:
            sym = sym[0] + '-'
    elif part_ind == 1 and len(sym) == 1 and sym[0] == 'b':
        sym = 'B'
    return sym


def add_min_to_lowercase(sym, part_ind):
    # only for the first part before '/', and not dominant chords
    not_minor_figures = ['x', 'o', 'h', 'd']
    minor = True
    for fig in not_minor_figures:
        if fig in sym:
            minor = False
            break
    if part_ind == 0 and sym.islower() and minor:
        # need to first get the lettername part
        # without the inversion and added notes
        sym += 'm'
    return sym


def replace_chord_with_dominant_seventh(sym):
    return sym.replace('x7', 'b7')


def preprocess_letters_before_sym2chord(sym):
    parts = sym.split('/')
    parts = [replace_flat_dash_primary(part, i) for i, part in enumerate(parts)]
    parts = [add_min_to_lowercase(part, i) for i, part in enumerate(parts)]
    parts = [replace_chord_with_dominant_seventh(part) for part in parts]

    return '/'.join(parts)


def sym2chord(sym, transpose=0):
    ch = None
    if is_roman_numeral(sym):
        try:
            chord_sym = roman.RomanNumeral(sym)
            # transpose
            chord_sym.transpose(transpose, inPlace=True)
            pches = chord_sym.pitches

            if chord_sym.secondaryRomanNumeral is not None:
                ch = chord.Chord(pches).transpose(-12)
            else:
                ch = chord.Chord(pches)
        except:
            print 'WARNING: symbol not found', sym

    else:
        try:
            sym = preprocess_letters_before_sym2chord(sym)

            chord_sym = harmony.ChordSymbol(sym)
            ch = chord.Chord(chord_sym.pitches).transpose(12)
        except:
            print 'WARNING: symbol not found', sym

    return ch


def roman2letter(sym):
    # print 'roman2letter', sym
    chord_sym = sym2roman(sym)
    if chord_sym is None:
        # return sym, False
        return None
    ch = chord.Chord(chord_sym.pitches)
    lettername, ch_type = harmony.chordSymbolFigureFromChord(ch, True)
    # print lettername, ch_type
    return lettername


def tokenize_transpose(sym, transpose):
    return '%s_%s' % (sym, transpose)


def roman2letter_transpose(sym, transpose=0,
                           translation_dict={},
                           return_dict=False):
    # print 'roman2letter', sym
    if sym in ROMAN_RELABELS.keys():
        sym = ROMAN_RELABELS[sym]
    key = tokenize_transpose(sym, transpose)
    if key in translation_dict:
        return translation_dict[key]
    rn = roman.RomanNumeral(sym)
    rn.transpose(transpose, inPlace=True)
    pitches = rn.pitches
    # pitches = [pch.midi+transpose for pch in rn.pitches]
    ch = chord.Chord(pitches)
    # print ch
    # ch = sym2chord(sym, transpose=transpose)
    if ch is None:
        # return sym, False
        return None
    lettername, ch_type = harmony.chordSymbolFigureFromChord(ch, True)

    # somehow music21 is not able to cope with
    ## ch = roman2letter_transpose('V', transpose=-1)
    # if lettername == 'Chord Symbol Cannot Be Identified':
    #     rn.transpose(transpose, inPlace=True)
    #     ch = chord.Chord(rn.pitches)
    #     lettername, ch_type = harmony.chordSymbolFigureFromChord(ch, True)

    # from resource.py
    for co_replacements in LETTER_PARTIAL_CO_RELABELS:
        all_in = True
        for replacements in co_replacements:
            if replacements[0] not in lettername:
                all_in = False
        if all_in:
            for replacements in co_replacements:
                lettername = lettername.replace(replacements[0],
                                                replacements[1])

    for k, v in LETTER_PARTIAL_RELABELS_FOR_USER.iteritems():
        if k in lettername:
            lettername = lettername.replace(k, v)

    translation_dict[key] = lettername
    # print lettername#, ch_type
    if return_dict:
        return lettername, translation_dict
    return lettername


def letter2roman(sym):
    # for now assume in key of C
    print '--- letter2roman', sym,
    try:
        chord_sym = harmony.ChordSymbol(sym)
    except:
        print 'WARNING: chord symbol does not exist'
        return None
    ch = chord.Chord(chord_sym.pitches)
    rn = roman.romanNumeralFromChord(ch, key.Key('C'))
    print rn.figure
    return rn.figure


def syms2score(syms):
    score = stream.Stream()
    for sym in syms:
        score.append(sym2chord(sym))
    return score


# def syms2score(syms):
#     score = stream.Stream()
#     for sym in syms:
#         rn = roman.RomanNumeral(sym)
#         rn.lyric = sym
#         score.append(rn)
#     return score


if __name__ == '__main__':
    # lettername = roman2letter('ii/o6/5')
    # lettername = roman2letter('It6')
    # print lettername
    ch = roman2letter_transpose('V', transpose=-1)
    print ch
    ch = roman2letter_transpose('VII6', transpose=-1)
    print ch
