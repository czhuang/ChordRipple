
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import value_and_grad
from autograd.util import quick_grad_check


def obj_func(weight):
    val = npr.normal()
    print 'random sample:', val
    return val*weight


def test():
    weight = 2.0
    obj_func(weight)
    training_loss_and_grad = value_and_grad(obj_func)

    print training_loss_and_grad(weight)


if __name__ == '__main__':
    test()


