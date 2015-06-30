

import numpy as np


def get_trans(v1, v2, trans):
    return - np.log(trans[v1[1], v2[1]])


def get_probs(v, probs):
    return - np.log(probs[v[1]])


def simple_foward_backward_gap_dist(model, before_sym, after_sym):
    # TODO: gap size only one for now
    syms = model.syms
    trans = model.trans
    if before_sym not in syms and after_sym not in syms:
        return None, None

    probs = np.zeros((len(syms,)))
    if before_sym is not None:
        before_ind = syms.index(before_sym)
    if after_sym is not None:
        after_ind = syms.index(after_sym)
    for i, sym in enumerate(syms):
        if after_sym is None:
            probs[i] = np.log(trans[before_ind, i])
        elif before_sym is None:
            probs[i] = np.log(trans[i, after_ind])
        else:
            probs[i] = np.log(trans[before_ind, i]) + np.log(trans[i, after_ind])
    sorted_inds = np.argsort(-probs)
    sorted_probs = [probs[ind] for ind in sorted_inds]
    sorted_syms = [syms[ind] for ind in sorted_inds]
    return sorted_probs, sorted_syms


def shortest_path(model, fixed, seq_ind, original_seq):
    syms = model.syms
    lb = np.min(fixed.keys())
    ub = np.max(fixed.keys())

    if seq_ind - 1 in fixed.keys():
        fixed[seq_ind-1] = []
    if seq_ind + 1 in fixed.keys():
        fixed[seq_ind+1] = []

    changed_start_ind = 0
    changed_end_ind = None
    sym_inds = range(lb, ub+1)

    # setting up the vertices
    states = {}  # for retriving the vertices according to time step
    vertices = []  # flat list of all vertices
    for i in range(np.min(fixed.keys()), ub+1):
        if i in fixed.keys() and len(fixed[i]) > 0:
            local_states = [(i, syms.index(fixed[i]))]
        else:
            local_states = [(i, j) for j in range(len(syms))]
        states[i] = local_states
        vertices.extend(local_states)

    dists = []
    first_states = states[lb]
    probs = model.priors
    if lb == 0:
        probs = model.starts
    for v in first_states:
        dists.append(get_probs(v, probs))
    for i in range(len(first_states), len(vertices)):
        dists.append(np.inf)
    assert len(vertices) == len(dists)

    previous_dict = {}
    for v in vertices:
        previous_dict[v] = None

    open_vertices = vertices[:]
    open_dists = dists[:]
    while len(open_vertices) > 0:
        ui = np.argmin(open_dists)
        u = open_vertices[ui]
        open_vertices.remove(u)
        dist = open_dists[ui]
        del open_dists[ui]

        state_ind = u[0]
        n_state_ind = state_ind + 1
        if n_state_ind > ub:
            break
        else:
            n_vertices = states[n_state_ind]
            for nv in n_vertices:
                alt = dist + get_trans(u, nv, model.trans)
                ni = vertices.index(nv)
                if alt < dists[ni]:
                    dists[ni] = alt
                    # update open dists
                    if nv in open_vertices:
                        open_pi = open_vertices.index(nv)
                        open_dists[open_pi] = alt
                    previous_dict[nv] = u

    # find target
    targets = states[ub]
    last_step_dists = []
    for v in targets:
        ind = vertices.index(v)
        last_step_dists.append(dists[ind])
    min_ind = np.argmin(last_step_dists)
    u = targets[min_ind]

    # construct seq
    seq = []
    while previous_dict[u] is not None:
        seq.insert(0, u)
        u = previous_dict[u]
    seq.insert(0, u)

    sym_seq = []
    for s in seq:
        sym = syms[s[1]]
        sym_seq.append(sym)
    sym_subseq = sym_seq[changed_start_ind:changed_end_ind]

    assert len(sym_inds) == len(sym_subseq)

    # make complete seq
    if ub + 1 > len(original_seq) - 1:
        new_seq = original_seq[:lb] + sym_subseq
    else:
        new_seq = original_seq[:lb] + sym_subseq + original_seq[ub+1:]
    return new_seq, sym_inds
