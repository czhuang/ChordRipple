
from music21 import roman, stream, chord, harmony


def is_roman_numeral(sym):
    if 'V' in sym or 'I' or 'v' in sym or 'i' in sym:
        return True
    return False


def sym2roman(sym):
    try:
        chord_sym = roman.RomanNumeral(sym)
        print chord_sym
    except:
        print "WARNING: chord symbol not valid", sym
        chord_sym = None
    return chord_sym


def sym2chord(sym):
    if is_roman_numeral(sym):
        chord_sym = roman.RomanNumeral(sym)
        ch = chord.Chord(chord_sym.pitches)
    else:
        chord_sym = harmony.ChordSymbol(sym)
        ch = chord.Chord(chord_sym.pitches).transpose(12)
    return ch


def syms2score(syms):
    score = stream.Stream()
    for sym in syms:
        score.append(sym2chord(sym))
    return score


def syms2score(syms):
    score = stream.Stream()
    for sym in syms:
        rn = roman.RomanNumeral(sym)
        rn.lyric = sym
        score.append(rn)
    return score




