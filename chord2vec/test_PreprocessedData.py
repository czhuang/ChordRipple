
from PreprocessedData import *
from config import get_configs
from load_songs_tools import get_raw_data


def test_PreprocessedSeq():
    configs = get_configs()
    seqs, syms = get_raw_data(configs)
    window = configs["window"]
    seq = seqs[0]
    seq[1] = 'a'
    seq_p = PreprocessedSeq(seq, syms, configs["window"])
    data = PreprocessedData(seqs[:2], syms, window)
