
from copy import copy
import numpy as np


def compute_set_diversity(syms, model):
    vecs = [model.get_vec(sym) for sym in syms]
    ref_vec = np.ones((model.W1.shape[1]))
    unit_ref_vec = ref_vec / np.linalg.norm(ref_vec)
    thetas = []
    for i, vec in enumerate(vecs):
        if vec is None:
            thetas.append(None)
        else:
            unit_vec = vec/np.linalg.norm(vec)
            cosine = np.dot(unit_vec, unit_ref_vec)
            theta = np.arccos(cosine)
            thetas.append(theta)
    thetas = [theta for theta in thetas if theta is not None]
    thetas_sigma = np.std(thetas)
    return thetas_sigma


def compute_seq_diversity(model, seqs, picked_seqs, firstn):
    lens = [len(seq) for seq in seqs] + [len(seq) for seq in picked_seqs]
    if np.min(lens) < firstn:
        firstn = np.min(lens)
    seq_simgas = []
    for seq in seqs:
        new_set = copy(picked_seqs)
        new_set.append(seq)
        seq_sigma = []
        # lens = [len(seq) for seq in new_set]

        for i in range(firstn):
            sets = [seq[i] for seq in new_set]
            sigma = compute_set_diversity(sets, model)
            seq_sigma.append(sigma)
        seq_simgas.append(np.mean(seq_sigma))
    ind = np.argmax(seq_simgas)
    return seqs[ind], ind


def get_diverse_seqs(seqs, model, topn=5, firstn=7):
    if isinstance(seqs, list):
        seqs = copy(seqs)
    else:
        # than would be data format
        seqs = copy(seqs.subseqs)
    ind = np.random.random_integers(len(seqs)-1)
    picked_seqs = [seqs[ind]]
    seqs.pop(ind)
    for i in range(topn-1):
        pick_seq, ind = compute_seq_diversity(model, seqs, picked_seqs, firstn)
        assert seqs[ind] == pick_seq
        seqs.pop(ind)
        picked_seqs.append(pick_seq)
    return picked_seqs


def make_diverse_seqs(seq_model, diversity_model, len_, n):
    # make random sequence of len firstn
    seqs = [seq_model.gen_seq(len_) for _ in range(n)]
    return get_diverse_seqs(seqs, diversity_model, n, len_)


def test_make_diverse_seqs():
    from retrieve_model_tools import retrieve_NGram, retrieve_SkipGramNN
    seq_model = retrieve_NGram()
    diversity_model = retrieve_SkipGramNN()
    print seq_model.syms[:10]
    seqs = make_diverse_seqs(seq_model, diversity_model, 2, 8)
    print seqs


if __name__ == '__main__':
    test_make_diverse_seqs()
