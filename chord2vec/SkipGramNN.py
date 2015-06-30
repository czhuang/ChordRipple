
import cPickle as pickle

import numpy as np
import pylab as plt

import kayak as ky

from WeightsContainer import WeightsContainer
# from PreprocessedData import PreprocessedData, PreprocessedSeq


class SkipGramNN(object):
    def __init__(self, data=None, configs=None, fname=None):
        # TODO: fix hacky load vs init, caused by not including original seqs
        # TODO: some functions should not be used when retrieved without data
        if fname is not None:
            weights = self.load(fname)
        else:
            print '...SkipGramNN, data', type(data)
            self.data = data
            self.syms = data.syms

            self.configs = configs

            self.log_prior = self.compute_prior()

        self.weights = WeightsContainer(configs['random_scale'])

        self.Xs, self.Ts = {}, {}

        layer1_sz = configs['layer1_sz']
        syms_sz = len(data.syms)
        batch_sz = configs['batch_sz']

        self.batchers = {}
        for key in data.keys:
            self.batchers[key] = ky.Batcher(batch_sz, data.inputs[key].shape[0])
            print key, data.inputs[key].shape[0]
            self.Xs[key] = ky.Inputs(data.inputs[key], self.batchers[key])
            self.Ts[key] = ky.Targets(data.outputs[key], self.batchers[key])

        self.W1 = self.weights.new((syms_sz, layer1_sz), 'W1')
        self.B1 = self.weights.new((1, layer1_sz), 'B1')

        self.H1s, self.W2s, self.B2s, self.Ys = {}, {}, {}, {}
        self.loss = 0
        for key in data.keys:
            self.H1s[key] = ky.MatMult(self.Xs[key], self.W1) + self.B1
            self.W2s[key] = self.weights.new((layer1_sz, syms_sz), 'W2s_%s' % key)
            self.B2s[key] = self.weights.new((1, syms_sz), 'B2s_%s' % key)
            self.Ys[key] = ky.LogSoftMax(ky.MatMult(self.H1s[key], self.W2s[key]) + self.B2s[key])

            if hasattr(data, 'weights'):
                gram_weights = ky.Parameter(self.data.weights[key])
                self.loss += ky.MatMult(gram_weights,
                                        ky.LogMultinomialLoss(self.Ys[key], self.Ts[key]))
            else:
                assert False, 'ERROR: should have weights'
                self.loss += ky.MatSum(ky.LogMultinomialLoss(self.Ys[key], self.Ts[key]))

        if configs['regularize']:
            self.loss += self.weights.L1Norm

        if fname is not None:
            self.init_weights(weights)

    def compute_prior(self):
        # empirical prior: P(C)
        # i.e. if window=2, seq positions: A, B, C, D, E
        # skipgram joint P(A, B, C, D, E) = P(C) P(A|C) P(B|C) P(D|C) P(E|C)
        counts = np.ones((len(self.data.syms),))
        for values in self.data.inputs.values():
            counts += np.sum(values, axis=0)
        prior = counts/np.sum(counts)
        assert np.allclose(np.sum(prior), 1.0)
        return np.log(prior)

    def save(self, fname):
        save_dict = {'syms': self.syms,
                     'configs': self.configs,
                     'log_prior': self.log_prior,
                     'weights': self.weights,
                     'keys': self.data.keys}
        with open(fname, 'wb') as p:
            pickle.dump(save_dict, p)

        embedding_dict = {'syms': self.syms,
                          'W1': self.W1.value,
                          'configs': self.configs}
        encoding = 'rn'
        if self.configs['use_letternames']:
            encoding = 'letter'
        fname = 'embedding-%s-%s-%d-%d.pkl' % (self.configs['corpus'],
                                               encoding,
                                               self.configs['layer1_sz'],
                                               self.configs['max_iter'])

        # fname = 'embedding-' + fname
        with open(fname, 'wb') as p:
            pickle.dump(embedding_dict, p)

    def load(self, fname):
        with open(fname, 'rb') as p:
            save_dict = pickle.load(p)
        for key, val in save_dict.iteritems():
            setattr(self, key, val)
        keys = save_dict["keys"]

        # make dummy data
        class DummyData(object):
            def __init__(self, keys):
                n = 2  # data count
                # data weights
                self.weights = [np.random.random(n)]
                self.inputs = {}
                self.outputs = {}
                for key in keys:
                    self.inputs[key] = [np.random.random(n)]
                    self.outputs[key] = [np.random.random(n)]
        self.data = DummyData(keys)
        # return model weights
        return save_dict["weights"]

    def init_weights(self, weights, loss_ref=None):
        print '--- init weights ---'
        print weights.shape
        print self.weights.value.shape
        self.weights.value = weights
        if loss_ref is not None:
            assert np.allclose(self.compute_loss(), loss_ref)

    def retrieve_weights(self):
        fname = self.configs.name_prefix
        with open(fname, 'rb') as p:
            configs = pickle.load(p)
            weights = pickle.load(p)
            loss = pickle.load(p)
        self.init_weights(weights, loss)

    def pickle_weights(self):
        fname = self.configs.name_prefix
        with open(fname, 'wb') as p:
            pickle.dump(self.configs, p)
            pickle.dump(self.weights.value, p)
            pickle.dump(self.compute_loss(), p)

    def get_vec(self, sym):
        if sym not in self.syms:
            return None
        ind = self.syms.index(sym)
        return self.W1.value[ind, :]

    def train(self):
        opt_algorithm = self.configs['opt_algorithm']
        if opt_algorithm == 'sgd':
            model_loss = self.train_sgd()
        elif opt_algorithm == 'cg':
            model_loss = self.train_cg()
        else:
            assert False, 'ERROR: Algorithm %s not implemented' % opt_algorithm
        return model_loss

    def train_sgd(self):
        num_epochs = self.configs['num_epochs']
        learn_rate = self.configs['learn_rate']
        print '--- training ---'
        losses = []
        for epoch in xrange(num_epochs):
            ordering = np.random.permutation(np.arange(len(self.data.keys)))
            loss = 0
            for i in ordering:
                skipgram_loss = 0
                for batcher in self.batchers[self.data.keys[i]]:
                    skipgram_loss += self.loss.value
                    self.weights.value -= learn_rate * self.weights._d_out_d_self(self.loss)
                    # print_vector(self.weights.value, 'weights')
                print 'skipgram %s: loss: %.2f' % (self.data.keys[i], skipgram_loss)
                loss += skipgram_loss
            print '\t\t\tEpoch %d: loss: %.2f' % (epoch, loss)
            losses.append(loss[0])
            self.plot_w1(iter_num=epoch)

            # termination criteria
            convergence_diff = 1.0
            diff_loss = losses[-2] - losses[-1]
            print '\t\t\tDiff in loss: %.2f' % diff_loss
            if epoch != 0 and diff_loss < convergence_diff:
                print '=== Reached convergence! (loss diff < %.2f' % convergence_diff
                break
            self.loss_curve = losses
        return self.compute_loss(self.data)

    def train_cg(self, log=False):
        for key in self.data.keys:
            self.batchers[key].test_mode()
        print 'start loss: %.3f' % self.loss.value

        self.loss_curve = []

        def loss_func(xs):
            self.weights.value = xs
            obj_val = self.loss.value
            self.loss_curve.append(obj_val[0])
            return obj_val

        def grads_func(xs):
            self.weights.value = xs
            grads = self.weights._d_out_d_self(self.loss)
            return grads

        initial_xs = self.weights.value

        max_iter = self.configs["max_iter"]

        print '...running fmin_cg'
        import scipy.optimize as spo
        xs, fopt, func_calls, grad_calls, warnflag,\
            all_vecs = spo.fmin_cg(loss_func, initial_xs, grads_func,
                                   maxiter=max_iter, full_output=1, retall=1)

        print xs.shape
        print self.W1.value.shape
        print 'fopt', fopt
        print 'func_calls', func_calls
        print 'grad_calls', grad_calls
        print 'warn_flag', warnflag
        print 'all_vecs', len(all_vecs)
        assert np.allclose(xs, all_vecs[-1])

        print 'len of loss list: %d' % len(self.loss_curve)
        # print 'losses', self.loss_curve

        loss = self.loss.value
        print 'loss: %.3f' % loss
        self.weights.value = xs
        print '--- training (using CG) ---'
        computed_loss = self.compute_loss(self.data)

        if log:
            fname = 'w1-allvecs-%s.pkl' % self.configs.name
            print len(fname), fname
            with open(fname, 'wb') as p:
                pickle.dump(self.W1.value, p)
                pickle.dump(self.data.syms, p)
                pickle.dump(all_vecs, p)
                pickle.dump(computed_loss, p)
                pickle.dump(self.weights.value, p)

        # TODO: loss diff due to??
        print 'diff in loss: %.2f' % np.abs(loss - computed_loss)
        assert np.abs(loss - computed_loss) < 1.0

        return computed_loss

    def compute_loss(self, data=None):
        if data is None:
            data = self.data
        print '=== checking after train ==='
        Ys = {}
        for key in data.keys:
            self.Xs[key].data = data.inputs[key]
            self.Ts[key].data = data.outputs[key]
            self.batchers[key].test_mode()
            Ys[key] = self.Ys[key].value
            summed_Ys = np.sum(np.exp(Ys[key]), axis=1)
            assert np.allclose(summed_Ys, np.ones(summed_Ys.shape))
        print 'loss: %.2f' % self.loss.value
        return self.loss.value[0]

    def check_loss(self):
        loss = 0
        for key in self.data.keys:
            inputs = self.data.inputs[key]
            outputs = self.data.outputs[key]
            H1 = np.dot(inputs, self.W1.value) + self.B1.value
            Y = np.dot(H1, self.W2s[key].value) + self.B2s[key].value
            maxes = np.max(Y, axis=1, keepdims=True)
            Y -= np.log(np.sum(np.exp(Y-maxes), axis=1, keepdims=True)) + maxes
            summed_Ys = np.sum(np.exp(Y), axis=1)
            assert np.allclose(summed_Ys, np.ones(summed_Ys.shape))
            if hasattr(self.data, 'weights'):
                weights = np.asarray(self.data.weights[key])
                loss += -np.dot(weights, np.sum(Y * outputs, axis=1))
            else:
                loss += -np.sum(Y * outputs)
        print '=== check loss ==='
        print loss
        print 'loss: %.2f' % loss
        self.compute_cross_entropy()
        return loss

    def compute_cross_entropy(self):
        loss = 0
        for key in self.data.keys:
            inputs = self.data.inputs[key]
            outputs = self.data.outputs[key]
            H1 = np.dot(inputs, self.W1.value) + self.B1.value
            Y = np.dot(H1, self.W2s[key].value) + self.B2s[key].value
            maxes = np.max(Y, axis=1, keepdims=True)
            Y -= np.log(np.sum(np.exp(Y-maxes), axis=1, keepdims=True)) + maxes
            summed_Ys = np.sum(np.exp(Y), axis=1)
            assert np.allclose(summed_Ys, np.ones(summed_Ys.shape))
            loss += -np.sum(np.log2(np.exp(np.sum(Y * outputs, axis=1))))
        N = 0
        for inputs in self.data.inputs.values():
            N += len(inputs)
        print 'N: %d' % N
        loss /= N
        print '=== cross entropy ==='
        print loss
        print 'loss: %.2f' % loss
        return loss

    @staticmethod
    def score_topn(predictions, targets, top_n=2):
        # assert predictions.keys() == targets.keys()
        scores = {}
        for key, predicts in predictions.iteritems():
            predict_inds = np.argsort(-predicts, axis=1)[:, :top_n]
            target_inds = np.argmax(targets[key], axis=1)
            local_scores = []
            for i in range(predict_inds.shape[0]):
                topn_inds = predict_inds[i, :]
                if target_inds[i] in topn_inds:
                    local_scores.append(1)
                else:
                    local_scores.append(0)
            scores[key] = np.mean(local_scores)
        print '--- scores (top %d) ---' % top_n
        averaged_scores = []
        for key, score in scores.iteritems():
            print '%s: %.3f' % (key, score)
            averaged_scores.append(score)
        print '\t\t\toverall: %.3f' % np.mean(averaged_scores)
        return scores

    @staticmethod
    def score(predictions, targets):
        print predictions.keys()
        print targets.keys()
        # assert predictions.keys() == targets.keys()
        scores = {}
        for key, predicts in predictions.iteritems():
            predict_inds = np.argmax(predicts, axis=1)
            target_inds = np.argmax(targets[key], axis=1)
            scores[key] = np.mean(predict_inds == target_inds)
        print '--- scores ---'
        averaged_scores = []
        for key, score in scores.iteritems():
            print '%s: %.3f' % (key, score)
            averaged_scores.append(score)
        print '\t\t\toverall: %.3f' % np.mean(averaged_scores)
        return scores

    def predict(self, data, return_score=False):
        predictions = {}
        for key in data.keys:
            self.Xs[key].data = data.inputs[key]
            self.batchers[key].test_mode()
            predictions[key] = self.Ys[key].value
        if not return_score:
            return predictions
        else:
            scores = self.score(predictions, data.outputs)
            scores_topn = self.score_topn(predictions, data.outputs)
            return predictions, scores

    def predict_identity(self):
        predictions = {}
        for key in self.data.keys:
            self.Xs[key].data = np.identity(len(self.data.syms))
            self.batchers[key].test_mode()
            predictions[key] = self.Ys[key].value
        return predictions

    def predict_reverse(self, data, top_n=1, return_probs=False, return_all=False):
        # if isinstance(data, PreprocessedData):
        #     seqs = data.seqs_processed
        #     syms = data.syms
        #     assert data.syms == self.data.syms
        # elif isinstance(data, PreprocessedSeq):
        #     seqs = [data]
        #     syms = self.data.syms
        # elif isinstance(data, list):
        #     seqs = data
        #     syms = self.data.syms

        # print 'predict_reverse', type(data)
        # print data.__dict__.keys()

        # TODO: only allow one prediction at a time for now
        if hasattr(data, 'seqs_processed'):
            seqs = data.seqs_processed
            syms = data.syms
            assert data.syms == self.data.syms
        elif hasattr(data, 'inputs') and hasattr(data, 'outputs'):
            # only one pair of input output to predict
            seqs = [data]
            syms = self.syms
        else:
            assert False, 'ERROR: Data type not yet implemented.'

        log_predictions = self.predict_identity()
        # log_prior = self.log_prior.copy()
        log_prior = np.log(np.ones((len(syms)))/len(syms))
        scores = []
        log_posts = []
        log_posts_all = []
        for j, seq in enumerate(seqs):
            local_scores = np.zeros((len(seq.inputs),))
            local_log_posts = []
            local_log_posts_all = []
            for i in range(len(seq.inputs)):
                target_ind = seq.inputs[i]
                if target_ind is None:
                    log_post = None
                    local_log_posts.append(log_post)
                    local_log_posts_all.append([None])
                    continue
                # i.e. seq positions: A, B, C, D, E
                # joint_probs = P(C, A=a, B=b, D=d, E=e)
                log_joint_probs = log_prior.copy()
                for offset in seq.valid_offsets[i]:
                    key = str(offset)
                    if key not in self.data.keys:
                        continue
                    ind = seq.outputs[key][i]
                    # i.e. P(A=a|C)
                    log_joint_probs += log_predictions[key][:, ind]
                predicted_inds = np.argsort(-log_joint_probs)[:top_n]
                local_scores[i] = target_ind in predicted_inds
                # normalize for posterior
                norm = np.log(np.sum(np.exp(log_joint_probs)))
                log_post = log_joint_probs[target_ind] - norm
                log_post_all = log_joint_probs - norm
                local_log_posts.append(log_post)
                local_log_posts_all.append(log_post_all)
            scores.append(local_scores)
            log_posts.append(local_log_posts)
            log_posts_all.append(local_log_posts_all)
            print len(seq.inputs), len(local_log_posts)
            assert len(seq.inputs) == len(local_log_posts)

        mean_scores = [np.mean(score) for score in scores]
        mean_score_total = np.mean(mean_scores)
        print '=== top %d ===' % top_n
        print 'scores: %.3f' % mean_score_total

        mean_scores = [np.mean(score[2:]) for score in scores]
        mean_score = np.mean(mean_scores)
        print 'scores (without first, 2nd): %.3f' % mean_score

        first_scores = [score[0] for score in scores]
        print 'first score: %.3f' % np.mean(first_scores)

        second_scores = [score[1] for score in scores]
        print 'second score: %.3f' % np.mean(second_scores)

        filtered_log_posts = []
        for lps in log_posts:
            log_posts = [lp for lp in lps if lp is not None]
            filtered_log_posts.append(log_posts)

        posts = [np.sum(np.exp(log_post)) for log_post in filtered_log_posts]
        posts_total = np.sum(posts)
        print 'likelihood (posterior probs): %.3f' % posts_total
        if return_probs:
            if return_all:
                return log_posts, mean_score_total, log_posts_all
            else:
                return log_posts, mean_score_total
        else:
            if return_all:
                return posts_total, mean_score_total, log_posts_all
            else:
                return posts_total, mean_score_total

    @staticmethod
    def plot_seq_prob(seq, probs, configs, hold=False, label=''):
        n = 10
        if probs.size < n:
            n = probs.size
        probs = probs[:n]
        x = range(len(probs))
        plt.plot(x, probs, 'x-', label=label)
        plt.xticks(x, seq[:n])
        plt.title('sequence posterior probs')
        plt.ylabel('log probs')
        plt.legend()
        if not hold:
            header = 7 if len(seq) > 7 else len(seq)
            header_str = '_'.join(s for s in seq[:header])
            print 'header_str', header_str
            fname = 'posterior-%s-%s.pdf' % (header_str, configs.name)
            # fname = 'test.pdf'
            fname = fname.replace('/', '_')
            plt.savefig(fname)

    def plot_w1(self, annotate=False, highlight_syms=[], iter_num=None):
        plt.clf()
        configs = self.configs
        if iter_num is not None:
            fname_tag = 'iter_%d' % iter_num
        else:
            fname_tag = ''

        from plot_utilities import plot_vec
        # print self.W1.value.shape, len(self.data.syms)
        assert self.W1.value.shape[0] == len(self.data.syms)
        plot_vec(self.W1.value, self.data.syms, configs,
                 highlight_syms=highlight_syms,
                 doPCA=configs['do_pca'],
                 with_annotations=annotate,
                 fname_tag=fname_tag,
                 save=True)
        # plt.savefig('%s.pdf' % fname)

    def most_similar(self, positive=[], negative=[], topn=10):
        """
        adapted from https://github.com/piskvorky/gensim/blob/develop/gensim/models/word2vec.py
        """
        from six import string_types

        if isinstance(positive, string_types) and not negative:
            positive = [positive]

        # add weights for each word, if not already present;
        # default to 1.0 for positive and -1.0 for negative words
        positive = [(word, 1.0) if isinstance(word, string_types + (np.ndarray,))
                    else word for word in positive]
        negative = [(word, -1.0) if isinstance(word, string_types + (np.ndarray,))
                    else word for word in negative]

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if isinstance(word, np.ndarray):
                mean.append(weight * word)
            elif word in self.syms:
                ind = self.syms.index(word)
                mean.append(weight * self.W1.value[ind, :])
                all_words.add(ind)
            else:
                print "word '%s' not in vocabulary" % word
                return None
        if not mean:
            print "cannot compute similarity with no input"
            return None
        norm = np.linalg.norm(np.array(mean).mean(axis=0))  # .astype(REAL)
        mean = np.array(mean).mean(axis=0)/norm
        dists = np.dot(self.W1.value, mean)
        if not topn:
            return dists
        best = np.argsort(dists)[::-1][:topn + len(all_words)]
        # ignore (don't return) words from the input
        result = [(self.syms[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]
