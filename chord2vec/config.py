
import time

OPT_ALGORITHMS = ['cg', 'sgd']
CORPORA = ['bach', 'rock']


class Configs(dict):
    def __init__(self, layer1_sz, random_scale, regularize):
        dict.__init__(self)
        self['layer1_sz'] = layer1_sz
        self['random_scale'] = random_scale
        self['regularize'] = regularize

    @property
    def name_prefix(self):
        name_str = ''
        for key, item in self.iteritems():
            if isinstance(item, float):
                name_str += '%s_%.2f' % (key, item)
            elif isinstance(item, int):
                name_str += "%s_%d" % (key, item)
            else:
                name_str += "%s_%s" % (key, str(item))
            name_str += '-'
        len_limit = 150
        if len(name_str) > len_limit:
            return name_str[:len_limit]
        else:
            return name_str[:-1]

    @property
    def name(self):
        return self.name_prefix + '-' + self.get_timestamp()

    @staticmethod
    def get_timestamp():
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        return timestamp


def get_configs(print_config=True):
    random_scale = 1.0
    layer1_sz = 5  #2
    regularize = False

    configs = Configs(layer1_sz, random_scale, regularize)

    configs['retrieve_model'] = True

    configs['min_count'] = 1
    # configs['min_count'] = 20

    configs['window'] = 1
    configs['bigram'] = False
    configs['reduced_key'] = '1'  # for forward bigram
    # configs['reduced_key'] = '-1'  # for reverse prediction

    # algorithm for optimization
    # conjugate gradient or stochastic gradient descent
    configs['opt_algorithm'] = 'cg'
    assert configs['opt_algorithm'] in OPT_ALGORITHMS

    # setting for conjugate gradient
    configs['max_iter'] = 2000

    # setting for stochastic gradient descent
    configs['num_epochs'] = 30
    configs['learn_rate'] = 0.001
    configs['batch_sz'] = 256

    if layer1_sz > 2:
        configs['do_pca'] = True
    else:
        configs['do_pca'] = False

    # corpus setting
    # configs['corpus'] = 'bach'
    configs['corpus'] = 'rock'
    assert configs['corpus'] in CORPORA

    configs['augmented_data'] = False

    configs['use_letternames'] = True
    # configs['use_letternames'] = False

    # if configs['corpus'] == 'bach':
    #     configs['use_letternames'] = False

    configs['use_durations'] = False

    configs['use_original'] = True

    # not yet implemented in this version
    configs['duplicate_by_rotate'] = False

    if print_config:
        print configs.name

    return configs
