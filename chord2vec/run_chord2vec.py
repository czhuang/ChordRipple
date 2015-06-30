

from config import get_configs
from load_model_tools import get_data, get_models


def retrieve_skipgram_and_ngram():
    """
    By default, loads cached model.  To train new model, set configs['retrieve_model']
    as in the run_chord2vec function
    :return: neural-net skipgram, bigram model
    """
    configs = get_configs()
    data = get_data(configs)
    nn, bigram = get_models(data, configs)

    # saves a plot of the 2D PCA of the chord vectors
    nn.plot_w1()


def run_chord2vec():
    """
    To run the model.  To finishing the training early, set configs['max_iter'] to less.
    """
    configs = get_configs()
    configs['retrieve_model'] = False
    configs['max_iter'] = 1
    data = get_data(configs)
    nn, bigram = get_models(data, configs)
    print nn.plot_w1()


if __name__ == "__main__":
    # run_chord2vec()
    retrieve_skipgram_and_ngram()
    print 'Done'
