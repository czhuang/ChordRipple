

from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.util import quick_grad_check
from six.moves import range
from six.moves import zip

from neural_net_utilities import WeightsParser, make_batches, logsumexp


def make_nn_funs(V_size, layer_size, skip_keys, L2_reg):
    parser = WeightsParser()
    parser.add_weights(('weights', 1), (layer_size, V_size))
    parser.add_weights(('biases', 1), (1, V_size))

    parser.add_weights(('weights', 2), (V_size, layer_size))
    parser.add_weights(('biases', 2), (1, layer_size))

    def predictions(W_vect, X):
        """Outputs normalized log-probabilities."""
        # Bag of Words, all projection layers are shared,
        # and they are averaged so can collapse the dictionaries

        in_W = parser.get(W_vect, ('weights', 1))
        in_B = parser.get(W_vect, ('biases', 1))

        H = np.dot(X, in_W) + in_B
        out_W = parser.get(W_vect, ('weights', 2))
        out_B = parser.get(W_vect, ('biases', 2))
        out = np.dot(H, out_W) + out_B
        # Softmax
        out = out - logsumexp(out, axis=1)
        return out

    def cross_entropy(W_vect, X, T):
        lik = np.sum(predictions(W_vect, X) * T)
        return - lik / X.shape[0]

    def loss(W_vect, X, T):
        log_prior = -L2_reg * np.dot(W_vect, W_vect)
        log_lik = np.sum(predictions(W_vect, X) * T)
        return - log_prior - log_lik

    def frac_err(W_vect, X, T):
        frac_err = np.mean(np.argmax(T,  axis=1) !=
                           np.argmax(predictions(W_vect, X), axis=1))
        return frac_err

    return parser.N, predictions, loss, cross_entropy, frac_err


def dict_to_matrix(Xs):
    X = Xs.values()[skip_keys[0]].copy()
    for key in skip_keys[1:]:
        X = np.vstack((X, Xs[key]))
    return X


if __name__ == '__main__':
    # Network parameters
    # layer_sizes = [784, 200, 100, 10]
    BIGRAM = True
    OPT_TYPE = 'SGD'  # 'CG' # 'SGD'
    BATCH_MODE = False
    layer_size = 20  #100  #20
    L2_reg = 1.0  # 0.1

    # Training parameters
    param_scale = 0.1
    learning_rate = 1e-4  # 1e-3
    momentum = 0.3  # 0.3 (best so far), 0.5  # 0.9
    # batch_size = 256
    batch_size = 32  # 64
    num_epochs = 50  # 50

    # Load data
    from load_songs_tools import get_data
    data = get_data()
    print('...loaded data')
    Xs = data.inputs
    Ys = data.outputs
    skip_keys = Xs.keys()
    X = dict_to_matrix(Xs)
    for key in skip_keys[1:]:
        X = np.vstack((X, Xs[key]))

    Y = Ys.values()[skip_keys[0]].copy()
    for key in skip_keys[1:]:
        Y = np.vstack((Y, Ys[key]))

    test_data = data.get_test_data()
    X_test = test_data.inputs
    Y_test = test_data.outputs

    if BIGRAM:
        X = {'1': X['1']}
        Y = {'1': Y['1']}
        X_test = {'1': X_test['1']}
        Y_test = {'1': Y_test['1']}

    V_size = len(data.syms)
    print("V_size", V_size)
    assert X.keys() == Y.keys()
    assert X.values()[0].shape[1] == V_size

    # Make neural net functions
    N_weights, pred_fun, loss_fun, total_loss_fun, cross_entropy, frac_err = \
        make_nn_funs(V_size, layer_size, skip_keys, L2_reg)
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
        test_perf  = cross_entropy(W, X_test, Y_test)
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
        return train_perf


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
        best_train_perf = np.inf  # error
        best_W = None
        best_iter = None
        for epoch in range(num_epochs):
            perf = print_perf(epoch, W)
            if best_train_perf > perf:
                best_train_perf = perf.copy()
                best_W = W.copy()
                best_iter = epoch
            # learning_rate /= 2
            # print('learning_rate', learning_rate)
            for i in range(max_batch):
                local_grad = 0
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
        best_train_perf = 1
        num_epochs = 20
        for epoch in range(num_epochs):
            perf = print_perf(epoch, W)
            if best_train_perf > perf:
                best_train_perf = perf.copy()
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
        max_iter = 1
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

    print('===== best performance ======')
    print_perf(best_iter, best_W)
    cross_entropy = cross_entropy(best_W, X, Y)
    print('cross_entropy: {0:5}' % cross_entropy)
