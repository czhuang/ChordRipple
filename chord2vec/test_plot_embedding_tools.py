
from retrieve_model_tools import retrieve_SkipGramNN
from plot_embedding_tools import plot_vec


def test_plot():
    nn = retrieve_SkipGramNN()
    plot_vec(nn.W1, nn.syms)


if __name__ == '__main__':
    test_plot()
    print 'Done'
