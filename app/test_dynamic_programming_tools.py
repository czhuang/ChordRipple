

import numpy as np


from retrieve_model_tools import retrieve_NGram, retrieve_SkipGramNN
from dynamic_programming_tools import shortest_path


def test_continuations():
    # optimal ones...
    MAX_LEN = 6
    model = retrieve_NGram()
    start_chord = 'C'
    fixed = { 1: start_chord }
    fixed[MAX_LEN] = start_chord  # for looping around
    original_seq = [''] * (MAX_LEN+1)
    original_seq[0] = original_seq[-1] = start_chord
    # TODO: hack since shortest_path is for generating ripples.
    # it wants the ind for the middle chord,
    # and then unconstraints it's immediate surround chords
    seq_ind = int(MAX_LEN/2)
    seq, inds = shortest_path(model, fixed, seq_ind, original_seq)
    print seq
    print inds
    # ['C', 'C', 'F', 'C', 'F', 'C', 'F', 'C']
    # ['C', 'C', 'F', 'G', 'C', 'F', 'C']
    return seq


def test_sample_continuations():
    # rejection sampling
    MAX_LEN = 6
    model = retrieve_NGram()
    start_chord = 'C'
    original_seq = [start_chord]
    seqs = []
    num_samples = 30
    for i in range(num_samples):
        seq = model.gen_continuation(original_seq, MAX_LEN)
        if seq[-1] == start_chord:
            seqs.append(seq)
    print '# of seqs:', len(seqs)
    # only got 7 out of 30 that ends in C
    for seq in seqs:
        print seq
    return seqs


def test_choosing_between_multiple_alternatives():
    model = retrieve_NGram()
    first = ['C', 'Cm', 'C/E']
    second = ['F', 'Am', 'F/A']
    third = ['G', 'G7']

    first = ['C/E', 'Cm']
    second = ['Am', 'F/A']
    third = ['G7', 'Gsus4']

    fixed = {0: first, 1: second, 2: third}
    seq, inds = shortest_path(model, fixed)
    print seq


def get_ranking(sims, seq):
    rankings = []
    for i, sym in enumerate(seq):
        if i in sims:
            if sym not in sims[i]:
                rankings.append(None)
            else:
                rankings.append(sims[i].index(sym))
    return rankings


def print_max_min(sim_scores_dict):
    for sims_scores in sim_scores_dict.values():
        scores = [sim_score[1] for sim_score in sims_scores]
        # print np.min(scores), np.max(scores)


def test_all_sim_ripples():
    nn = retrieve_SkipGramNN()
    model = retrieve_NGram()

    original_seq = ['C/E', 'E/G#', 'G']
    the_same = []
    for i in range(20):
        original_seq = model.gen_seq(5)
        print '-------------------------'
        print 'orig seq\t', original_seq
        sims_dict = {}
        sims_scores_dict = {}
        for j in range(1, 4):
            sym = original_seq[j]
            sims_scores = nn.most_similar(sym, topn=30)
            # print sims_scores
            sims_scores_dict[j] = sims_scores
            # TODO: are the scores probabilities?
            sims = [sims[0] for sims in sims_scores]
            # print sym, sims
            sims_dict[j] = sims

        fixed = {}
        fixed_scores = {}
        for key, sims in sims_dict.iteritems():
            fixed[key] = sims
            fixed_scores[key] = sims_scores_dict[key]

        # fixed the middle chord
        fixed[2] = original_seq[2]
        fixed_scores[2] = original_seq[2]

        fixed[0] = fixed_scores[0] = original_seq[0]
        fixed[4] = fixed_scores[4] = original_seq[4]

        # for key, values in fixed.iteritems():
        #     print key, values

        seq, inds = shortest_path(model, fixed)
        # print original_seq
        print 'sim w/o seq\t', seq, get_ranking(sims_dict, seq)

        # print 'taking sim into consideration'
        seq_s, inds = shortest_path(model, fixed_scores)
        # print original_seq
        print 'sim w/score\t', seq_s, get_ranking(sims_dict, seq_s)

        seq_trans, inds = shortest_path(model, fixed, 2, original_seq)
        print 'just trans\t', seq_trans, get_ranking(sims_dict, seq_trans)
        print_max_min(sims_scores_dict)
        if seq == seq_s:
            the_same.append(0)
        else:
            # assert False
            the_same.append(1)
    print '# of different outcomes out of 20:', np.sum(the_same)


def test_all_sim_ripples_edited():
    nn = retrieve_SkipGramNN()
    model = retrieve_NGram()

    original_seq = model.gen_seq(5)
    original_seq = ['F', 'E-', 'F', 'C', 'F']
    original_seq = ['Fmaj7/C', 'G', 'B-', 'C', 'Em']
    original_seq = ['C', 'C', 'F', 'E-', 'Cm']
    # Cm, Gm, Cm
    # Cm, Eb, Ab
    print 'orig seq\t', original_seq

    fixed = {}
    fixed[0] = original_seq[0]
    fixed[4] = original_seq[4]
    fixed[2] = original_seq[2]

    seq, inds = shortest_path(model, fixed, 2, original_seq, nn)
    print 'sim w/score\t', seq

    seq, inds = shortest_path(model, fixed, 2, original_seq)
    print 'just trans\t', seq


if __name__ == '__main__':
    # test_continuations()
    # test_sample_continuations()
    # test_choosing_between_multiple_alternatives()
    # test_all_sim_ripples()
    test_all_sim_ripples_edited()