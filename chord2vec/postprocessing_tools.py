
import numpy as np

from sklearn.decomposition import PCA


def pca_project(vecs, n_components=2, whiten=False):
    pca = PCA(n_components=n_components)
    vecs_projected = pca.fit_transform(vecs)
    print '=== PCA projection ==='
    print pca.explained_variance_ratio_
    print 'choosen explained: %.2f' % np.sum(pca.explained_variance_ratio_[:n_components])
    return vecs_projected
