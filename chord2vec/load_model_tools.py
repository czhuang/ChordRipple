
import cPickle as pickle

import os

import numpy as np
import pylab as plt

from plot_utilities import plot_vec, add_arrow_annotation, make_song_dict
from config import Configs, get_configs
from SkipGramNN import SkipGramNN
from NGram import NGram
from load_songs_tools import get_data


def make_test_NGram():
    from load_songs_tools import load_songs
    from config import get_configs
    configs = get_configs()
    seqs = load_songs(configs)[:30]
    ngram = NGram(seqs, 2)
    return ngram


def make_NGram():
    from load_songs_tools import load_songs
    from config import get_configs
    configs = get_configs()
    seqs = load_songs(configs)
    ngram = NGram(seqs, 2)
    return ngram


class WrappeSkipGram(SkipGramNN):
    def __init__(self, weights, syms):
        self.W1 = weights
        self.syms = syms

    def get_vec(self, sym):
        if sym not in self.syms:
            return None
        ind = self.syms.index(sym)
        return self.W1[ind, :]


def get_proxy_model():
    fname = None
    print fname
    assert fname is not None, "Error: no model to retrieve in the time being"
    with open(fname, 'rb') as p:
        w1 = pickle.load(p)
        syms = pickle.load(p)
        model_loss = pickle.load(p)
        model_weights = pickle.load(p)
        # configs_reloaded = pickle.load(p)

    COMMON_TONE_CIRCLE = ['I', 'iii', 'V', 'viio6', 'ii', 'IV', 'vi', 'I']
    circle_name = COMMON_TONE_CIRCLE
    song_dict = make_song_dict(circle_name)

    configs_dummy = get_configs()
    fname, ax, vecs = plot_vec(w1, syms, configs_dummy, save=True,
                               doPCA=True, return_ax=True)
    add_arrow_annotation(syms, vecs, song_dict, ax, False)
    plt.savefig('w1-%s.pdf' % Configs.get_timestamp())

    model = WrappeSkipGram(w1, syms)
    return model


def get_models(data=None, configs=None, save=False):
    if configs is None:
        configs = get_configs()

    if data is None:
        data = get_data(configs)

    # TODO: remove hack
    if configs['bigram']:
        reduced_keys = [configs['reduced_key']]
        data.keys = reduced_keys
        test_data = data.get_test_data()
        test_data.keys = reduced_keys

    retrieve_model = configs['retrieve_model']
    model = SkipGramNN(data, configs)
    print 'SkipGramNN, # of syms', len(model.syms)

    if not retrieve_model:
        model_loss = model.train()
        if save:
            model.save('skipgram-%s.pkl' % (configs['corpus']))
            plt.clf()
            plt.plot(model.loss_curve)
            plt.savefig('losses-%s.png' % configs.name)
        print '=== train loss ==='
        print 'loss: %.2f' % model_loss
        loss = model.check_loss()
        if not configs['regularize']:
            assert np.allclose(loss, model_loss)

        if save:
            model_weights = model.weights.value
            fname = 'w1-%s.pkl' % configs.name
            print fname
            with open(fname, 'wb') as p:
                pickle.dump(model.W1.value, p)
                pickle.dump(data.syms, p)
                pickle.dump(model_loss, p)
                pickle.dump(model_weights, p)
                pickle.dump(configs, p)

            fname = 'skipgram-bach.pkl'
            model.save(fname)
    else:
        fname = os.path.join('data', 'test_skipgram_model.pkl')
        print fname
        assert fname is not None, "Error: no model to retrieve in the time being"
        with open(fname, 'rb') as p:
            w1 = pickle.load(p)
            syms = pickle.load(p)
            model_loss = pickle.load(p)
            model_weights = pickle.load(p)
            configs_reloaded = pickle.load(p)
        for key in configs.keys():
            if key not in configs_reloaded.keys():
                print 'no key', key
        for key in configs.keys():
            if key in configs_reloaded.keys():
                if configs[key] != configs_reloaded[key]:
                    print configs[key], configs_reloaded[key]

        # assert configs == configs_reloaded
        model.init_weights(model_weights, model_loss)

    train_seq_data = data.get_train_seqs_data()
    train_seqs = [seq for seq in train_seq_data.seqs]
    syms = data.syms

    # ngram_model = NGram(train_seqs, syms, 2, configs)
    ngram_model = NGram(data.seqs, syms, 2, configs)
    print '\n\nNgram, # of syms', len(ngram_model.syms)
    if save:
        ngram_model.save('bigram-%s.pkl' % (configs['corpus']))
    print len(ngram_model.syms), len(model.data.syms)
    assert ngram_model.syms == model.data.syms

    return model, ngram_model


if __name__ == '__main__':
    # get_weights_syms()
    get_models()
