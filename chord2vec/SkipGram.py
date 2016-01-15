from __future__ import absolute_import
from __future__ import print_function

import os
import cPickle as pickle

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.util import quick_grad_check
from six.moves import range
from six.moves import zip

from autograd_utilities import WeightsParser, make_batches, logsumexp


def make_nn_funs(V_size, layer_size, skip_keys, L2_reg):
    parser = WeightsParser()

    parser.add_shape(('weights', 1), (V_size, layer_size))
    parser.add_shape(('biases', 1), (1, layer_size))

    for key in skip_keys:
        parser.add_shape(('weights', 2, key), (layer_size, V_size))
        parser.add_shape(('biases', 2, key), (1, V_size))

    def predictions(W_vect, X, key):
        """Outputs normalized log-probabilities."""
        in_W = parser.get(W_vect, ('weights', 1))
        in_B = parser.get(W_vect, ('biases', 1))

        H = np.dot(X, in_W) + in_B
        out_W = parser.get(W_vect, ('weights', 2, key))
        out_B = parser.get(W_vect, ('biases', 2, key))
        out = np.dot(H, out_W) + out_B
        # Softmax
        out = out - logsumexp(out, axis=1)
        return out

    def cross_entropy(W_vect, X, T):
        cross_entropy = 0
        N = 0
        for key in X.keys():
            cross_entropy += np.sum(predictions(W_vect, X[key], key) * T[key])
            N += X[key].shape[0]
        cross_entropy = - cross_entropy / N
        # print('Cross entropy: %.5f' % cross_entropy)
        return cross_entropy

    def loss(W_vect, X, T, key):
        log_prior = -L2_reg * np.dot(W_vect, W_vect)
        log_lik = np.sum(predictions(W_vect, X, key) * T)
        return - log_prior - log_lik

    def total_loss(W_vect, Xs, Ts):
        log_prior = -L2_reg * np.dot(W_vect, W_vect)
        loss = - log_prior
        for key in Xs.keys():
            loss -= np.sum(predictions(W_vect, Xs[key], key) * Ts[key])
        # print('loss: 0:5, previous: 1:5' % (loss, previous_loss))
        # previous_loss = loss
        return loss

    def frac_err(W_vect, Xs, Ts):
        # TODO: by taking in a dictionary, not consistent with other functions
        # sometimes takes in a dictionary with a subset of keys
        # hence use Xs.keys() instead of skipgram keys
        frac_err = 0
        for key in Xs.keys():
            frac_err += np.mean(np.argmax(Ts[key], axis=1) !=
                                np.argmax(predictions(W_vect, Xs[key], key), axis=1))
        return frac_err / len(Xs.keys())

    print(parser.idxs_and_shapes)
    return parser, predictions, loss, total_loss, cross_entropy, frac_err

if __name__ == '__main__':
    # Network parameters
    # layer_sizes = [784, 200, 100, 10]
    BIGRAM = True
    USE_ALL_DATA_FOR_TRAINING = True

    OPT_TYPE = 'CG'  #'SGD'  #'CG'

    BATCH_MODE = False  # False meanings use Stoachastic GD
                        # True means batch everything in one
    layer_size = 1  #2  #20  # 30  #20  #10  #100  #20
    window = 1  #2

    # Training parameters
    L2_reg = 0.01  #0.001  #0.001  #0.01 (was reasonable for SGD)  #0  # 0.1
    param_scale = 0.1
    learning_rate = 1e-3  # 1e-4  # 1e-3
    momentum = 0.3  # 0.3 (best so far), 0.5  # 0.9

    batch_size = 128
    # batch_size = 32  #64  # 1024  # 256  # 32  # 64

    # for 'SGD'
    num_epochs = 1000  #20  #30  # 30 # 50

    # for 'CG'
    max_iter = 500


    # Load data
    from load_songs_tools import get_data
    data = get_data()
    if not USE_ALL_DATA_FOR_TRAINING:
        X = data.inputs
        Y = data.outputs

        test_data = data.get_test_data()
        X_test = test_data.inputs
        Y_test = test_data.outputs
    else:
        print('==== USING ALL DATA FOR BOTH TRAIN AND TEST ====')
        X = data.inputs_all
        Y = data.outputs_all

        X_test = data.inputs_all
        Y_test = data.outputs_all

    print('...loaded data')
    print('# of keys:', len(X.keys()))
    print('# of datapoints for training:')
    N = X.values()[0].shape[0]
    for key in X.keys():
        print(key, X[key].shape)

    assert len(X.keys()) == window * 2

    print('# of datapoints for testing:')
    for key in X_test.keys():
        print(key, X_test[key].shape)

    if BIGRAM:
        X = {'1': X['1']}
        Y = {'1': Y['1']}
        X_test = {'1': X_test['1']}
        Y_test = {'1': Y_test['1']}

    skip_keys = X.keys()
    V_size = len(data.syms)
    print("V_size", V_size)
    assert X.keys() == Y.keys()
    assert X.values()[0].shape[1] == V_size

    # Make neural net functions
    parser, pred_fun, loss_fun, total_loss_fun, cross_entropy, frac_err = \
        make_nn_funs(V_size, layer_size, skip_keys, L2_reg)
    N_weights = parser.num_weights
    print('N_weights', N_weights)
    print

    # Initialize weights
    rs = npr.RandomState()
    W = rs.randn(N_weights) * param_scale

    # Check the gradients numerically, just to be safe
    for key in skip_keys:
        quick_grad_check(loss_fun, W, (X[key], Y[key], key))

    # print("    Epoch      |    Train err  |   Test error  ")
    print("    Epoch      |    Train cross|   Test cross  ")

    def print_perf(epoch, W):
        train_perf = cross_entropy(W, X, Y)
        if not USE_ALL_DATA_FOR_TRAINING:
            test_perf  = cross_entropy(W, X_test, Y_test)
        else:
            test_perf = train_perf
        X_test_forward = {'1': X_test['1']}
        Y_test_forward = {'1': Y_test['1']}
        test_perf_just_forward  = cross_entropy(W, X_test_forward, Y_test_forward)

        if not BIGRAM:
            X_test_backward = {'-1': X_test['-1']}
            Y_test_backward = {'-1': Y_test['-1']}
            test_perf_just_backward  = cross_entropy(W, X_test_backward, Y_test_backward)

        # train_perf = frac_err(W, X, Y)
        # test_perf  = frac_err(W, X_test, Y_test)
        # X_test_forward = {'1': X_test['1']}
        # Y_test_forward = {'1': Y_test['1']}
        # test_perf_just_forward  = frac_err(W, X_test_forward, Y_test_forward)
        #
        # if not BIGRAM:
        #     X_test_backward = {'-1': X_test['-1']}
        #     Y_test_backward = {'-1': Y_test['-1']}
        #     test_perf_just_backward  = frac_err(W, X_test_backward, Y_test_backward)

        # print("{0:15}|{1:15}|{2:15}".format(epoch, train_perf, test_perf))
        if not BIGRAM:
            print("{0:15}|{1:15}|{2:15}|{3:15}|{4:15}".format(epoch, train_perf,
                                                       test_perf,
                                                       test_perf_just_forward,
                                                       test_perf_just_backward))
        else:
            print("{0:15}|{1:15}|{2:15}|{3:15}".format(epoch, train_perf,
                                                       test_perf,
                                                       test_perf_just_forward))
        return test_perf


    if OPT_TYPE == 'SGD' and not BATCH_MODE:
        loss_grad = grad(loss_fun)
        # Train with sgd
        batch_idxs = [make_batches(X[key].shape[0], batch_size)
                      for key in skip_keys]
        cur_dir = np.zeros(N_weights)

        idx_tracker = [0] * len(skip_keys)
        num_batches = [len(idxs) for idxs in batch_idxs]
        max_batch = np.max(num_batches)

        reduced_count = 0
        best_test_perf = np.inf  # error
        best_W = None
        best_iter = None
        for epoch in range(num_epochs):
            perf = print_perf(epoch, W)
            if best_test_perf > perf:
                best_test_perf = perf.copy()
                best_W = W.copy()
                best_iter = epoch
            # learning_rate /= 2
            # print('learning_rate', learning_rate)
            for i in range(max_batch):
                skip_keys_order = np.random.permutation(len(skip_keys))
                for ki in skip_keys_order:
                    if i < num_batches[ki]:
                        idxs = batch_idxs[ki][i]
                        key = skip_keys[ki]
                        grad_W = loss_grad(W, X[key][idxs], Y[key][idxs], key)
                        cur_dir = momentum * cur_dir + (1.0 - momentum) * grad_W
                        #local_grad += cur_dir
                        # print('local gradient', i, key, np.dot(local_grad, local_grad))
                        # W -= learning_rate * local_grad
                        W -= learning_rate * cur_dir
    elif OPT_TYPE == 'SGD' and BATCH_MODE:
        # try SGD on total_loss_fun
        loss_grad = grad(total_loss_fun)
        best_test_perf = 1
        num_epochs = 20
        for epoch in range(num_epochs):
            perf = print_perf(epoch, W)
            if best_test_perf > perf:
                best_test_perf = perf.copy()
                best_W = W.copy()
                best_iter = epoch
            grad_W = loss_grad(W, X, Y)
            W -= learning_rate * grad_W

    elif OPT_TYPE == 'CG':
        start_loss = total_loss_fun(W, X, Y)
        print('...check loss for CG')
        quick_grad_check(total_loss_fun, W, (X, Y))

        print('start_loss', start_loss)
        loss_grad = grad(total_loss_fun)
        print('...running fmin_cg')
        import scipy.optimize as spo

        best_W, fopt, func_calls, grad_calls, warnflag,\
            all_vecs = spo.fmin_cg(total_loss_fun, W, loss_grad, args=(X, Y),
                                   maxiter=max_iter, full_output=1,
                                   retall=1)
        # just to give a number since can't retrieve iteration number
        best_iter = func_calls
        print_perf(best_iter, best_W)
        print(best_W.shape)
        print('fopt', fopt)
        print('warn_flag', warnflag)
    else:
        assert False, 'ERROR: not yet implemented.'

    print('===== best test performance ======')
    print_perf(best_iter, best_W)
    cross_entropy = cross_entropy(best_W, X_test, Y_test)
    print('cross_entropy:', cross_entropy)

    # store the best weights
    if OPT_TYPE == 'SGD':
        max_iter = num_epochs
    fname = 'window-%d_bigram-%s_hiddenSize-%d_crossEntropy-%.3f_bestIter-%d-maxIter-%d_opt-%s_l2reg-%.4f_batchSize-%d_momemtum-%.2f_N-%d_V-%d.pkl' \
            % (window, str(BIGRAM), layer_size, cross_entropy, best_iter,
               max_iter, OPT_TYPE, L2_reg, batch_size, momentum, N, V_size)
    fpath = os.path.join('models', 'rock-letter', 'chord2vec', fname)
    print(fpath)
    results = dict(parser=parser, W=best_W, syms=data.syms)
    results['iter'] = best_iter
    results['cross_entropy'] = cross_entropy

    with open(fpath, 'wb') as p:
        pickle.dump(results, p)



