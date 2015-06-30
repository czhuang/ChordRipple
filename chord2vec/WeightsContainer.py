

import numpy as np
import numpy.random as npr

import kayak as ky


class WeightsContainer(object):
    def __init__(self, random_scale):
        self.N = 0
        self._weights_list = []
        self._names_list = []

        self.random_scale = random_scale

    def new(self, shape=None, name="", init_val=None):
        assert shape is not None or init_val is not None
        if init_val is not None:
            w_new = ky.Parameter(init_val)
            shape = init_val.shape
        else:
            w_new = ky.Parameter(self.random_scale*npr.randn(*shape))
        self._weights_list.append(w_new)
        self.N += np.prod(shape)
        self._names_list.append(name)
        return w_new

    def _d_out_d_self(self, out):
        grad_list = [out.grad(w) for w in self._weights_list]
        return np.concatenate([arr.ravel() for arr in grad_list])

    def params(self, name):
        ind = self._names_list.index(name)
        return self._weights_list[ind]

    @property
    def value(self):
        return np.concatenate([w.value.ravel() for w in self._weights_list])

    @value.setter
    def value(self, vect):
        vect = vect.ravel()
        for w in self._weights_list:
            sub_vect, vect = np.split(vect, [np.prod(w.shape)])
            w.value = sub_vect.reshape(w.shape)

    @property
    def L1Norm(self):
        l1norm_weight = 0.5
        l1norms = None
        for w in self._weights_list:
            l1norm = ky.L1Norm(w, l1norm_weight)
            if l1norms is None:
                l1norms = l1norm
            else:
                l1norms = ky.ElemAdd(l1norms, l1norm)
        return l1norms

    def init_momentum(self):
        self._momentum = np.zeros((self.N,))

    @property
    def momentum(self):
        return self._momentum

    @momentum.setter
    def momentum(self, vect):
        self._momentum = vect