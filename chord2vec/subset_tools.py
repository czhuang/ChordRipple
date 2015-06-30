

import numpy as np


def get_circle_vecs(W1, syms, circle_chords=None,
                    rn=False, major=True):
    # collect raw circle vectors
    if circle_chords is None:
        chords = ['C', 'G', 'D', 'A', 'E', 'B', 'Gb', 'Db',
                  'Ab', 'Eb', 'Bb', 'F']
    else:
        chords = circle_chords
    if not rn:
        if not major:
            chords = [c.lower() for c in chords]
    inds = [syms.index(c) for c in chords]
    X = [W1[ind, :] for ind in inds]
    X = np.asarray(X)
    return X, chords


def subset_pca(W1, syms, subset_chords, n_components=2):

    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)

    X, chords = get_circle_vecs(W1, syms, circle_chords=subset_chords,
                                rn=True)
    print X.shape
    pca.fit(X)
    W1_projected = pca.transform(W1)

    print '=== PCA projection ==='
    print pca.explained_variance_ratio_
    print 'choosen explained: %.2f' % np.sum(pca.explained_variance_ratio_)

    return W1_projected


def subset(mat, syms, row_syms, col_syms):
    mat_subset = np.zeros((len(row_syms), len(col_syms)))
    col_inds = [syms.index(sym) for sym in col_syms]
    for i, row_sym in enumerate(row_syms):
        ind = syms.index(row_sym)
        mat_subset[i, :] = mat[ind, col_inds]
    return mat_subset
