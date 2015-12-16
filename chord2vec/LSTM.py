""" adapted from autograd's lstm.py for rock chords """

from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import value_and_grad
from autograd.util import quick_grad_check
from scipy.optimize import minimize
#from builtins import range

import os
import cPickle as pickle
import copy

from autograd_utilities import WeightsParser
from autograd_utilities import sigmoid, activations, logsumexp
from utility_tools import retrieve_most_recent_fname

from Data import Data

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTM(object):
    def __init__(self, data, state_size, train_iters,
                 retrieve_weights, pretrain):
        npr.seed(1)

        self.state_size = state_size
        self.train_iters = train_iters

        self.param_scale = 0.01

        self.data = data
        self.train_inputs = data.train_inputs

        self.parser, self.num_weights = self.build_lstm()

        self.training_loss, self.training_loss_and_grad, = self.make_loss_funs()

        self.path = os.path.join("models", "rock-letter", "lstm")

        self.retrieve_weights = retrieve_weights
        self.pretrain = pretrain
        self.init_weights = self.init_params(retrieve_weights, pretrain)

    def build_lstm(self):
        """Builds functions to compute the output of an LSTM."""
        input_size = self.data.input_size
        output_size = self.data.output_size
        state_size = self.state_size

        parser = WeightsParser()
        parser.add_shape('init_cells',   (1, state_size))
        parser.add_shape('init_hiddens', (1, state_size))
        parser.add_shape('change',  (input_size + state_size + 1, state_size))
        parser.add_shape('forget',  (input_size + 2 * state_size + 1, state_size))
        parser.add_shape('ingate',  (input_size + 2 * state_size + 1, state_size))
        parser.add_shape('outgate', (input_size + 2 * state_size + 1, state_size))
        parser.add_shape('predict', (state_size + 1, output_size))
        return parser, parser.num_weights

    def update_lstm(self, input, hiddens, cells, forget_weights, change_weights,
                    ingate_weights, outgate_weights):
        """One iteration of an LSTM layer."""
        change  = np.tanh(activations(change_weights, input, hiddens))
        forget  = sigmoid(activations(forget_weights, input, cells, hiddens))
        ingate  = sigmoid(activations(ingate_weights, input, cells, hiddens))
        cells   = cells * forget + ingate * change
        outgate = sigmoid(activations(outgate_weights, input, cells, hiddens))
        hiddens = outgate * np.tanh(cells)
        return hiddens, cells

    def hiddens_to_output_probs(self, predict_weights, hiddens):
        output = activations(predict_weights, hiddens)
        return output - logsumexp(output)     # Normalize log-probs.

    def predict(self, weights, inputs):
        """Outputs normalized log-probabilities of each character, plus an
           extra one at the end."""
        parser = self.parser
        forget_weights  = parser.get(weights, 'forget')
        change_weights  = parser.get(weights, 'change')
        ingate_weights  = parser.get(weights, 'ingate')
        outgate_weights = parser.get(weights, 'outgate')
        predict_weights = parser.get(weights, 'predict')
        num_sequences = inputs.shape[1]
        hiddens = np.repeat(parser.get(weights, 'init_hiddens'), num_sequences, axis=0)
        cells   = np.repeat(parser.get(weights, 'init_cells'),   num_sequences, axis=0)

        output = [self.hiddens_to_output_probs(predict_weights, hiddens)]
        for input in inputs:  # Iterate over time steps.
            hiddens, cells = self.update_lstm(input, hiddens, cells, forget_weights,
                                              change_weights, ingate_weights, outgate_weights)
            output.append(self.hiddens_to_output_probs(predict_weights, hiddens))
        return output

    def log_likelihood(self, weights, inputs, targets):
        logprobs = self.predict(weights, inputs)
        loglik = 0.0
        num_time_steps, num_examples, _ = inputs.shape
        for t in range(num_time_steps):
            loglik += np.sum(logprobs[t] * targets[t])
        return loglik / (num_time_steps * num_examples)

    def make_loss_funs(self):
        # Wrap function to only have one argument, for scipy.minimize.
        def training_loss(weights):
            targets = self.data.get_targets(self.train_inputs)
            return -self.log_likelihood(weights, self.train_inputs, targets)

         # Build gradient of loss function using autograd.
        training_loss_and_grad = value_and_grad(training_loss)
        return training_loss, training_loss_and_grad

    def predict_conditional(self, weights, seeds, seq_len):
        """ Outputs both normalized log-probabilities and also max predicted seq
            based on some seed (inputs)"""
        parser = self.parser
        forget_weights  = parser.get(weights, 'forget')
        change_weights  = parser.get(weights, 'change')
        ingate_weights  = parser.get(weights, 'ingate')
        outgate_weights = parser.get(weights, 'outgate')
        predict_weights = parser.get(weights, 'predict')
        num_sequences = seeds.shape[1]
        hiddens = np.repeat(parser.get(weights, 'init_hiddens'), num_sequences, axis=0)
        cells   = np.repeat(parser.get(weights, 'init_cells'),   num_sequences, axis=0)

        output = [self.hiddens_to_output_probs(predict_weights, hiddens)]
        # the 2nd dimension is the num of seq
        seed_len = seeds.shape[1]
        output_seq = np.empty((1, 1, seeds.shape[2]))
        for i in range(seq_len):
            if i >= seed_len:
                input = output_seq[i]
            else:
                input = seeds[i]
            hiddens, cells = self.update_lstm(input, hiddens, cells, forget_weights,
                                              change_weights, ingate_weights, outgate_weights)
            logprobs = self.hiddens_to_output_probs(predict_weights, hiddens)
            output.append(logprobs)
            sym = self.data.syms[np.argmax(logprobs, axis=1)]
            sym_encoded = self.data.string_to_one_hot([sym])
            input = self.data.get_targets(input)
            subseq = self.data.one_hot_to_string(input) + [sym]

            # TODO: when have multiple attributes might need to change
            attributes = self.data.compute_attributes(subseq)[-1:, np.newaxis]
            target = np.hstack((sym_encoded, attributes))

            output_seq = np.vstack((output_seq, target[:, np.newaxis, :]))

        output_seq = self.data.get_targets(output_seq)
        return output, output_seq

    def init_params(self, retrieve_weights, pretrain):
        assert not(pretrain and retrieve_weights)

        if retrieve_weights:
            fname = retrieve_most_recent_fname(self.path)
            assert fname is not None
            print('Retrieving weights from:', fname)
            with open(fname, 'rb') as p:
                store_dict = pickle.load(p)

            # check that they're from the same model architecture
            assert store_dict["state_size"] == self.state_size
            init_weights = store_dict["trained_weights"]
            assert init_weights.size == self.num_weights
            assert np.isclose(store_dict["train_loss"],
                              self.training_loss(init_weights))
            self.start_iter = store_dict["train_iters"]

        else:
            init_weights = npr.randn(self.num_weights) * self.param_scale
            self.start_iter = 0

        if pretrain:
            from retrieve_SkipGram_weights import retrieve_chord2vec_weights
            chord2vec_weights, chord2vec_biases, syms = retrieve_chord2vec_weights()
            print("Initializing some weights with chord2vec embeddings...")
            assert syms == self.data.syms

            # compare norms
            norms = np.linalg.norm(chord2vec_weights, axis=1)
            self.pretrain_scaler = self.param_scale * 5  #1.0
            chord2vec_weights = chord2vec_weights / norms[:, None] * self.pretrain_scaler
            print('median in embedding', np.mean(np.linalg.norm(chord2vec_weights, axis=1)))

            random_weights = self.parser.get(init_weights, 'change')
            random_norms = np.linalg.norm(random_weights, axis=1)
            print('median in random norms', np.mean(random_norms))

            self.parser.set(init_weights, 'change', chord2vec_weights)
            self.parser.set(init_weights, 'predict', chord2vec_weights)
        return init_weights

    def fit(self, callback_on):
        self.iter_count = self.start_iter
        max_iters = np.maximum(0, self.train_iters - self.start_iter)
        print("Remaining # of iters:", max_iters, "out of", self.train_iters)

        def callback(weights):
            self.iter_count += 1
            print("\nTrain loss for iter %d: %.4f" % (self.iter_count, self.training_loss(weights)))
            print_training_prediction(weights)

        def print_training_prediction(weights):
            print("Training text                         Predicted text")
            logprobs = np.asarray(self.predict(weights, self.train_inputs))
            # Just show first few examples
            for s in range(logprobs.shape[1])[:10]:
                training_text  = self.data.one_hot_to_string(self.train_inputs[:,s,:])
                predicted_text = self.data.one_hot_to_string(logprobs[:, s, :])
                print(' '.join(training_text) + "|" + ' '.join(predicted_text))

        # Check the gradients numerically, just to be safe
        print('init_weights', self.init_weights.shape)
        quick_grad_check(self.training_loss, self.init_weights)

        print("Training LSTM...")
        if callback_on:
            callback_fun = callback
        else:
            callback_fun = None
        print('change\t, forget, ingate, outgate, cells\t, hiddens\t')

        result = minimize(self.training_loss_and_grad, self.init_weights, jac=True, method='CG',
                          options={'maxiter': max_iters}, callback=callback_fun)
        self.trained_weights = result.x

        # storing for continuing training next time
        train_loss = result.fun
        assert np.isclose(train_loss, self.training_loss(self.trained_weights))

        store_dict = dict(state_size=state_size,
                          train_iters=train_iters,
                          trained_weights=self.trained_weights,
                          train_loss=train_loss)

        fpath = os.path.join(self.path, self.make_fname(store_dict))
        with open(fpath, 'wb') as p:
            pickle.dump(store_dict, p)

    def make_fname(self, store_dict):
        fname = 'stateSize-%d_trainIter-%d_seqLen-%d_trainLoss-%.4f_pretrain-%s' % (self.state_size,
                    self.train_iters, self.data.seq_length, store_dict["train_loss"],
                    str(self.pretrain))
        if self.pretrain:
            fname = '%s_scaler-%.4f' % (self.pretrain_scaler)

        fname += '_id-%d.pkl' % np.random.randint(1e+7)

        print(fname, fname)
        return fname

    def test_loss(self):
        data = self.data
        test_targets = self.data.get_targets(data.test_inputs)
        test_loss = -self.log_likelihood(self.trained_weights,
                                         data.test_inputs, test_targets)
        print("\nTest loss:", test_loss)
        return test_loss

    def generate_conditional(self, seed_seqs=None, seed_len=5):
        if seed_seqs is not None and isinstance(seed_seqs, list):
            seed_inputs = self.data.build_data_repr(seed_seqs)
        if seed_seqs is None:
            seed_inputs = self.data.test_inputs

        print("\nGenerating chords from seed to LSTM model...")
        seed_inputs = seed_inputs[:seed_len, :, :]
        desired_len = self.train_inputs.shape[0]
        num_seed_seq = seed_inputs.shape[1]
        for s in range(num_seed_seq):
            # print(' '.join(self.data.test_seqs[s][:seed_len]))
            seq = seed_inputs[:, s:s+1, :]

            # outputs can't be used directly,  unless [-1].ravel()
            outputs, output_seq = self.predict_conditional(self.trained_weights,
                                                           seq, desired_len)
            # assumes one sequence too
            output_strs = self.data.one_hot_to_string(output_seq[:, 0, :])
            seed_strs = self.data.one_hot_to_string(seq)
            print(' '.join(seed_strs))
            print(' '.join(output_strs[:seed_len]), ' | ', ' '.join(output_strs[seed_len:]))

    def generate_from_scratch(self, num_seqs=20):
        print("\nGenerating chords from LSTM model...")
        for t in range(num_seqs):
            text = []
            for i in range(seq_length):
                seqs = data.string_to_one_hot(text)[:, np.newaxis, :]

                # indices -1 because assuming only put in one sequence...
                logprobs = self.predict(lstm.trained_weights, seqs)[-1].ravel()
                sym = data.syms[npr.choice(len(logprobs), p=np.exp(logprobs))]
                text.append(sym)
            print(' '.join(text))


if __name__ == '__main__':
    seq_length = 15
    num_seq = -1  # -1
    state_size = 50  #100 # 20
    train_iters = 100  #200  #500 #500 #20  #500

    retreive_weights = True
    retrieve_data = True
    pretrain = False
    callback_on = False

    data = Data(seq_length, num_seq, retrieve_data)
    lstm = LSTM(data, state_size, train_iters, retreive_weights, pretrain)

    lstm.fit(callback_on)
    lstm.test_loss()

    lstm.generate_conditional()
    lstm.generate_from_scratch()

    # compare to bigram
    from benchmark_tools import ngram_generate_seqs
    # ngram_generate_seqs(data.train_seqs, data.syms)

