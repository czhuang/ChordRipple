
import autograd.numpy as np


class WeightsParser(object):
    """A helper class to index into a parameter vector."""
    def __init__(self):
        self.idxs_and_shapes = {}
        self.N = 0

    def add_weights(self, name, shape):
        start = self.N
        self.N += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.N), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)


def make_batches(N_total, batch_sz):
    start = 0
    batches = []
    while start < N_total:
        batches.append(slice(start, start + batch_sz))
        start += batch_sz
    return batches


def logsumexp(X, axis=1):
    max_X = np.max(X)
    # running_sum = np.sum(np.exp(X - max_X), axis=axis, keepdims=True)
    # print('max_X', max_X)
    # print('running_sum', running_sum)
    # print('max, min', np.max(X), np.min(X))
    return max_X + np.log(np.sum(np.exp(X - max_X), axis=axis, keepdims=True))