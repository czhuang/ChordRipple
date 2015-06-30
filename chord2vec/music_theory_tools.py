

USING_LETTERNAMES = True


def uppercase_chord(chord):
    if chord == 'b':
        upper = chord.upper()
    elif len(chord) > 1 and chord[0] == 'b' and chord[1] == 'b':
        upper = chord[0].upper() + chord[1:]
    elif chord[0] == 'b':
        upper = chord[0] + chord[1:].upper()
    elif len(chord) > 1 and chord[1] == 'b':
        upper = chord[0].upper() + chord[1:]
    else:
        upper = chord.upper()
    return upper


if USING_LETTERNAMES:
    ROMAN_NUMERAL_ORDERING = ['c', 'db', 'd', 'eb', 'e', 'f', 'gb', 'g', 'ab', 'a', 'bb', 'b']
else:
    ROMAN_NUMERAL_ORDERING = ['i', 'bii', 'ii', 'biii', 'iii', 'iv', 'bv', 'v', 'bvi', 'vi', 'bvii', 'vii']
    SIMPLE_CHORDS_ORDERED = ['VI', 'II', 'V', 'I', 'IV',
                             'bVII', 'bIII', 'bVI', 'bII', 'bV', 'VII', 'III',
                             'vi', 'ii', 'v', 'i', 'iv',
                             'bvii', 'biii', 'bvi', 'bii', 'bv', 'vii', 'iii']

if USING_LETTERNAMES:
    CIRCLE_OF_FIFTHS_MAJOR = ['A', 'D', 'G', 'C', 'F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb', 'B', 'E']
    # CIRCLE_OF_FIFTHS_MINOR = [ note.lower() for note in CIRCLE_OF_FIFTHS_MAJOR ]
    CIRCLE_OF_FIFTHS_MINOR = [note+'m' for note in CIRCLE_OF_FIFTHS_MAJOR]
else:
    CIRCLE_OF_FIFTHS_MAJOR = ['VI', 'II', 'V', 'I', 'IV',
                              'bVII', 'bIII', 'bVI', 'bII', 'bV', 'VII', 'III']
    CIRCLE_OF_FIFTHS_MAJOR = ['V/ii', 'II', 'V', 'I', 'IV',
                              'bVII', 'bIII', 'bVI', 'bII', 'V/vii', 'V7/iii', 'V/vi']

    CIRCLE_OF_FIFTHS_MINOR = ['vi', 'ii', 'v', 'i', 'iv',
                              'bvii', 'biii', 'bvi', 'bii', 'bv', 'vii', 'iii']


def make_circle_of_fifths_dict(circle_of_fifths_chords):
    cof_dict = {}
    for i, chord in enumerate(circle_of_fifths_chords[::-1]):
        ind = (i+1) % len(circle_of_fifths_chords)
        cof_dict[chord] = circle_of_fifths_chords[::-1][ind]
    return cof_dict
CIRCLE_OF_FIFTHS_MAJOR_DICT = make_circle_of_fifths_dict(CIRCLE_OF_FIFTHS_MAJOR)
CIRCLE_OF_FIFTHS_MINOR_DICT = make_circle_of_fifths_dict(CIRCLE_OF_FIFTHS_MINOR)
# print len(CIRCLE_OF_FIFTHS_MAJOR_DICT), CIRCLE_OF_FIFTHS_MAJOR_DICT
# print len(CIRCLE_OF_FIFTHS_MINOR_DICT), CIRCLE_OF_FIFTHS_MINOR_DICT

DESCENDING_THIRDS_MAJOR_KEY = ['I', 'vi', 'IV', 'ii', 'vii', 'V', 'iii', 'I']
DESCENDING_THIRDS_MINOR_KEY = ['i', 'VI', 'iv', 'ii', 'VII', 'V', 'III', 'i']


def make_progression_dict_from_chord_list(chords):
    cof_dict = {}
    for i, chord in enumerate(chords[:-1]):
        ind = (i+1) % len(chords)
        cof_dict[chord] = chords[ind]
    return cof_dict
DESCENDING_THIRDS_MAJOR_KEY_DICT = make_progression_dict_from_chord_list(DESCENDING_THIRDS_MAJOR_KEY)
# print DESCENDING_THIRDS_MAJOR_KEY_DICT
DESCENDING_THIRDS_MINOR_KEY_DICT = make_progression_dict_from_chord_list(DESCENDING_THIRDS_MINOR_KEY)
# print DESCENDING_THIRDS_MINOR_KEY_DICT


def get_relative_minor_dict():
    relative_minor = {}
    for i, chord in enumerate(ROMAN_NUMERAL_ORDERING):
        upper = uppercase_chord(chord)
        relative_minor[upper] = ROMAN_NUMERAL_ORDERING[i-3]
    return relative_minor
RELATIVE_MINOR = get_relative_minor_dict()
# print 'RELATIVE_MINOR'
# for key, item in RELATIVE_MINOR.iteritems():
#     print key, item
