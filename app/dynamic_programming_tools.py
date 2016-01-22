

import numpy as np

# WEIGHTING_BETWEEN_SIM_AND_TRANS
WEIGHT = 1.0


def is_list_or_tuple(v):
    if isinstance(v, list) or isinstance(v, tuple):
        return True
    else:
        return False


def get_idx(v):
    if is_list_or_tuple(v[1]):
        return v[1][0]
    else:
        return v[1]


def get_trans(v1, v2, trans):
    v1_idx = get_idx(v1)
    v2_idx = get_idx(v2)
    prob = - np.log(trans[v1_idx, v2_idx])
    # TODO: assumes that first one is accounted for earlier
    if is_list_or_tuple(v2[1]):
        prob += - np.log(v2[1][1]) * WEIGHT
    return prob


def get_probs(v, probs):
    v_idx = get_idx(v)
    prob = - np.log(probs[v_idx])
    if is_list_or_tuple(v[1]):
        # TODO: assumes accounting for first one
        prob += - np.log(v[1][1]) * WEIGHT
    return prob


def simple_foward_backward_gap_dist(model, before_sym, after_sym, experiment_type=None):
    # TODO: gap size only one for now
    syms = model.syms

    # print 'simple_foward_backward_gap_dist'
    # print 'syms', syms
    trans = model.trans
    if before_sym not in syms and after_sym not in syms:
        return None, None
    if before_sym is None and experiment_type == 0:
        return None, None

    assert np.allclose(np.sum(trans, axis=1), 1)

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


def has_single_sym(item):
    if isinstance(item, list) and len(item) == 1:
        return True
    elif isinstance(item, str) or isinstance(item, unicode):
        return True
    else:
        return False


def shortest_path(model, fixed, seq_ind=None, original_seq=None,
                  nn=None):
    print '--- shortest path for constructing ripple ---'
    print fixed
    # number of similarity matches to query on the neural net
    NN_TOPN = 30
    syms = model.syms
    lb = np.min(fixed.keys())
    ub = np.max(fixed.keys())

    if seq_ind is not None:
        idx = seq_ind - 1
        if idx in fixed.keys():
            fixed[idx] = []

        idx = seq_ind + 1
        # TODO: need to check upper bound?
        if idx in fixed.keys():
            fixed[idx] = []

    # print fixed
    # print 'lb, ub', lb, ub
    changed_start_ind = 0
    for i in range(lb, ub+1):
        if i in fixed and not has_single_sym(fixed[i]):
            changed_start_ind = i
            break

    changed_end_ind = ub
    for i in range(ub, lb-1, -1):
        if i in fixed and not has_single_sym(fixed[i]):
            changed_end_ind = i
            break

    #print changed_start_ind, changed_end_ind
    # sym_inds = range(lb, ub+1)
    sym_inds = range(changed_start_ind, changed_end_ind+1)
    if seq_ind is not None and seq_ind not in sym_inds:
        sym_inds.append(seq_ind)
        sym_inds.sort()


    print fixed, lb, ub
    # setting up the vertices
    states = {}  # for retriving the vertices according to time step
    vertices = []  # flat list of all vertices
    # TODO: need to streamline, right now allows sim scores to come in and also to be generated locally
    # for i in range(np.min(fixed.keys()), ub+1):
    for i in range(lb, ub+1):
        local_states = []
        if i in fixed.keys() and len(fixed[i]) > 0:
            syms_options = fixed[i]
            print 'syms_options', syms_options
            if isinstance(syms_options, list):

                for sym in syms_options:
                    if isinstance(sym, tuple):
                        local_states.append((i, (syms.index(sym[0]), sym[1])))
                    else:
                        local_states.append((i, syms.index(sym)))
            else:
                # assert if it's a tuple
                print 'not a tuple'
                assert not isinstance(syms_options, tuple)
                # print "syms", len(syms), syms
                local_states = [(i, syms.index(syms_options))]
        else:
            if nn is None:
                local_states = [(i, j) for j in range(len(syms))]
            else:
                sym = original_seq[i]
                if sym in syms:
                    sims_and_scores = nn.most_similar(sym, topn=NN_TOPN)
                    for sim_score in sims_and_scores:
                        local_states.append((i, (syms.index(sim_score[0]), sim_score[1])))
                    # print '...populating', i, len(local_states)
                elif len(sym) == 0:
                    for sym in syms:
                        local_states.append((i, syms.index(sym)))
                else:
                    print 'WARNING: %s not in list, not doing similarity query' % \
                        sym
                    local_states.append((i, syms.index(sym)))

        states[i] = local_states
        vertices.extend(local_states)

    # print states.keys()
    for key, values in states.iteritems():
        print key, len(values)

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

    #print 'seq', seq

    sym_seq = []
    for s in seq:
        # sym = syms[s[1]]
        sym = syms[get_idx(s)]
        sym_seq.append(sym)

    # start_offset = changed_start_ind - lb
    # end_offset = changed_end_ind - ub
    # sym_subseq = sym_seq[start_offset:end_offset]
    sym_subseq = sym_seq
    # print 'start_offset, end_offset', start_offset, end_offset
    #print sym_subseq

    #print len(sym_inds), len(sym_subseq)
    # assert len(sym_inds) + start_offset - end_offset == len(sym_subseq)

    # make complete seq
    if original_seq is None:
        return sym_subseq, range(len(sym_subseq))
    elif changed_end_ind + 1 > len(original_seq) - 1:
        # new_seq = original_seq[:changed_start_ind] + sym_subseq
        new_seq = original_seq[:lb] + sym_subseq
    else:
        # new_seq = original_seq[:changed_start_ind] + sym_subseq + \
        #           original_seq[changed_end_ind+1:]
        new_seq = original_seq[:lb] + sym_subseq + \
                  original_seq[ub+1:]

    # print 'original_seq', original_seq
    # print 'new_seq', new_seq
    # print 'sym_inds', sym_inds
    return new_seq, sym_inds
