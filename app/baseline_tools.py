

from retrieve_model_tools import retrieve_NGram

from music21 import harmony

# no D# and G# for now
ROOT_ORDER = ['C', 'C#', 'Db', 'D', 'Eb', 'E', 'F',
              'F#', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

CHORD_TYPE_ORDER = ['major', 'minor', 'augmented', 'diminished',
                    'dominant-seventh', 'major-seventh', 'minor-seventh',
                    'half-diminished-seventh', 'dimished-seventh', 'minor-major-seventh']


def order_by_root():
    model = retrieve_NGram()
    syms = model.syms

    root_dict = {}
    for sym in syms:
        root = harmony.root(sym)
        if root not in root_dict:
            root_dict.append(sym)
        root_dict[]



