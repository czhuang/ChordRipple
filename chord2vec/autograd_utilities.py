
import autograd.numpy as np


class WeightsParser(object):
    """A helper class to index into a parameter vector."""
    def __init__(self):
        self.idxs_and_shapes = {}
        self.num_weights = 0

    def add_shape(self, name, shape):
        start = self.num_weights
        self.num_weights += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.num_weights), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)

    def get_shape(self, name):
        return self.idxs_and_shapes[name][1]

    def set(self, vect, name, new_vect, offset=0):
        idxs, shape = self.idxs_and_shapes[name]
        if shape == new_vect.shape:
            vect[idxs] = np.ravel(new_vect)
        else:
            sub_vect = vect[idxs]
            assert sub_vect.size >= new_vect.size
            sub_vect[offset:offset+new_vect.size] = np.ravel(new_vect)


def make_batches(N_total, batch_sz):
    start = 0
    batches = []
    while start < N_total:
        batches.append(slice(start, start + batch_sz))
        start += batch_sz
    return batches


def sigmoid(x):
    return 0.5*(np.tanh(x) + 1.0)   # Output ranges from 0 to 1.


def activations(weights, *args):
    # concatenate the weights with biases? for each input dimension
    cat_state = np.concatenate(args + (np.ones((args[0].shape[0],1)),), axis=1)
    return np.dot(cat_state, weights)


def logsumexp(X, axis=1):
    max_X = np.max(X)
    return max_X + np.log(np.sum(np.exp(X - max_X), axis=axis, keepdims=True))


if __name__ == '__main__':
    parser = WeightsParser()
    parser.add_shape('w1', (3, 1))
    parser.add_shape('w2', (3, 3))

    weights = np.ones(parser.num_weights)

    # want to set w1
    # shape = parser.get_shape('w2')
    zeros = np.zeros((2, 3))

    parser.set(weights, 'w2', zeros, 3)

    w1 = parser.get(weights, 'w1')
    print 'w1', w1

    w2 = parser.get(weights, 'w2')
    print 'w2', w2