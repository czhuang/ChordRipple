
import numpy as np

from QueryObject import QueryObject
from load_model_tools import get_models
from load_songs_tools import get_configs_data

from retrieve_model_tools import retrieve_SkipGramNN, retrieve_NGram
from ngram import NGram
from music21_chord_tools import get_targets


def get_sim_chords(model, sym, topn=7):
    similars = model.most_similar(sym, topn=topn)
    # print 'similars', similars
    if similars is None:
        return
    sims = [s[0] for s in similars]
    print 'sims to', sym, ':', sims
    return sims

def get_diff_chords(model, sym, topn=7):
    similars = model.most_diff(sym, topn=topn)
    # print 'similars', similars
    if similars is None:
        return
    sims = [s[0] for s in similars]
    print 'diff to', sym, ':', sims
    return sims

def get_chord_

def get_novelty(model, seq):
    model.novelty(seq, 1, seq[1])

def get_typical_novel(model, subs, query, sub_type, topn=2):
    assert isinstance(model, NGram)
    threshold = 0.0003
    scores = []
    for sub in subs:
        seq_sub = query.sub(sub)
        scores.append(model.novelty(seq_sub))
    scores_normalized = np.asarray(scores) / np.sum(scores)

    print '--- most %s ---' % sub_type
    for sym, score in zip(subs, scores_normalized):
        print '%s: %.7f' % (sym, score)
    sorted = np.sort(scores_normalized)
    sorted_indices = np.argsort(scores_normalized)
    n = len(sorted_indices)
    indices_larger_than_threshold = np.arange(n)[sorted > threshold]
    min_ind = np.min(indices_larger_than_threshold)

    # if the # of subs is not enough to have
    # topn for each typical and novel side then just collect all from one side
    enough_for_both_ends = n - min_ind >= topn*2
    if enough_for_both_ends:
        start_ind = -topn
    else:
        start_ind = min_ind
    typical_novel_dict = {}
    # typical: the scores would be large since approxs joint
    key = '%s_typical' % sub_type
    inds = sorted_indices[start_ind:]
    syms = [subs[ind] for ind in inds]
    scores = [scores_normalized[ind] for ind in inds]
    typical_novel_dict[key] = [syms, scores]
    if enough_for_both_ends:
        key = '%s_novel' % sub_type
        inds = sorted_indices[min_ind:min_ind+topn]
        syms = [subs[ind] for ind in inds]
        scores = [scores_normalized[ind] for ind in inds]
        typical_novel_dict[key] = [syms, scores]

    return typical_novel_dict


def gen_sim_novelty_quandrant_chords(models, query):
    # sim (typical, novelty), relative
    # diff (typical, novelty), relative
    # two for each quandrant
    nn, ngram = models
    sims = get_sim_chords(nn, query.sym)
    diffs = get_diff_chords(nn, query.sym)

    quandrants = get_typical_novel(ngram, sims, query, 'similar')
    temp = get_typical_novel(ngram, diffs, query, 'different')
    quandrants.update(temp)
    for key, items in quandrants.iteritems():
        print key, items
    return quandrants


def sample_targets(model, query):
    # TODO: assumes Markov
    # TODO: if slow, for now sample for all symbols
    print '...sample_targets'
    sym = query.sym
    nsamples = 100  # 100
    steps = 5
    targets_all = {}
    for i in range(nsamples):
        current_sym = sym
        seq = [sym]
        for s in range(steps):
            next_sym = model.sample_next([current_sym])
            seq.append(next_sym)
        # check what targets are arrived at
        # print 'seq: ', seq
        targets = get_targets(seq)
        for key, counts in targets.iteritems():
            if key in targets_all:
                targets_all[key] += counts
            else:
                targets_all[key] = counts
    print '--- summed target counts ---'
    print targets_all
    return targets_all

    # TODO: "seq:  ['G', 'G', 'F', 'C', 'C', 'C']" could have have no 'C' count



def check_diff_chords():
    model = retrieve_SkipGramNN()
    get_sim_chords(model, 'C')
    get_diff_chords(model, 'C')

    get_sim_chords(model, 'F')
    get_diff_chords(model, 'F')

    get_sim_chords(model, 'F/A')
    get_diff_chords(model, 'F/A')

    # wanted to compare how chords after G and G7 might be different

    get_sim_chords(model, 'G')
    get_diff_chords(model, 'G')

    get_sim_chords(model, 'G7')
    get_diff_chords(model, 'G7')

    ngram = retrieve_NGram()
    ngram.next_top_n('G')
    ngram.next_top_n('G7')


def check_novelty():
    ngram = retrieve_NGram()
    seq = ['C', 'G', 'C']
    get_novelty(ngram, seq)
    seq = ['C', 'D-', 'C']
    get_novelty(ngram, seq)
    seq = ['C', 'G/B', 'C']
    get_novelty(ngram, seq)


def check_quandrant():
    nn = retrieve_SkipGramNN()
    ngram = retrieve_NGram()
    models = [nn, ngram]

    seq = ['C', 'F', 'G', 'C']
    query = QueryObject(seq, 2, 'G')
    gen_sim_novelty_quandrant_chords(models, query)


def check_sample_targets():
    ngram = retrieve_NGram()
    seq = ['C', 'F', 'A', 'C']
    # {'Fm': 0, 'C': 43, 'F/A': 0, 'Cm': 4, 'F': 0}
    query = QueryObject(seq, 2)
    sample_targets(ngram, query)


def check_get_targets():
    # seq = ['D', 'G', 'C', 'F', 'B-', 'E-',
    #        'A-', 'D-', 'F#', 'B', 'E', 'A', 'D']
    # seq.reverse()
    seq = ['D', 'G']
    # seq = ['D-', 'G', 'A']
    # seq = ['D-', 'G', 'A', 'D']
    seq = ['D-7', 'F#']
    get_targets(seq)


if __name__ == '__main__':
    # check_diff_chords()
    # check_novelty()
    # check_quandrant()
    check_sample_targets()
    # check_get_targets()

