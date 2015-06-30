import numpy as np
import kayak


# This file adapts some utility functions from https://github.com/piskvorky/gensim/


class Vocab(object):
    """
    from gensim word2vec:
    A single vocabulary item, used internally for constructing binary trees
    (incl. both word leaves and inner nodes).
    """

    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "<" + ', '.join(vals) + ">"


def _vocab_from(sentences):
    """
    adapted from gensim word2vec
    """
    sentence_no, vocab = -1, {}
    for sentence_no, sentence in enumerate(sentences):
        for word in sentence:
            if word in vocab:
                vocab[word].count += 1
            else:
                vocab[word] = Vocab(count=1)
    return vocab


def build_vocab(sentences, min_count):
    """
    Build vocabulary from a sequence of sentences (can be a once-only generator stream).
    Each sentence must be a list of unicode strings.

    """
    vocab = _vocab_from(sentences)
    # print 'vocab', vocab
    # assign a unique index to each word
    vocab2index, index2word = {}, []
    for word, v in vocab.iteritems():
        if v.count >= min_count:
            v.index = len(vocab2index)
            index2word.append(word)
            vocab2index[word] = v
    return vocab2index, index2word


def collect_valid_skip_grams(input_labels, output_labels_dict,
                             weights=None):
    from PreprocessedData import SymbolInt

    input_labels_valid_dict = {}
    output_labels_valid_dict = {}
    for key, output_labels in output_labels_dict.iteritems():
        local_input_labels = []
        local_output_labels = []
        for i, output_label in enumerate(output_labels):
            if output_label is not None:
                if weights is None:
                    local_input_labels.append(input_labels[i])
                    local_output_labels.append(output_label)
                else:
                    assert len(weights) == len(output_labels)
                    w = weights[i]
                    out_w = weights[i + int(key)]
                    if weights[i + int(key)] < w:
                        w = out_w
                    input_sym = SymbolInt(input_labels[i], w)
                    output_sym = SymbolInt(output_label, w)
                    local_input_labels.append(input_sym)
                    local_output_labels.append(output_sym)
        # check
        assert len(local_input_labels) == len(local_output_labels)
        input_labels_valid_dict[key] = local_input_labels
        output_labels_valid_dict[key] = local_output_labels
    return input_labels_valid_dict, output_labels_valid_dict


def build_skip_grams_complete(sentences, index2word, window):
    inputs = []
    outputs = {}
    total_words = np.sum([len(sentence) for sentence in sentences])
    print 'total_words', total_words
    print 'window', window

    for sen in sentences:
        for pos in range(window, len(sen) - window):
            # check if input is in index2word
            if sen[pos] not in index2word:
                continue

            # check if all the skipgram output positions are in index2word
            outputs_all_in = True
            for offset in range(-window, window + 1):
                output_word = sen[pos + offset]
                if output_word not in index2word:
                    outputs_all_in = False
            if not outputs_all_in:
                continue

            inputs.append(index2word.index(sen[pos]))
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                output = index2word.index(sen[pos + offset])
                key = str(offset)
                if key not in outputs.keys():
                    outputs[key] = [output]
                else:
                    outputs[key].append(output)

    for output in outputs.values():
        assert len(inputs) == len(output)
    return inputs, outputs


def build_skip_grams(sentences, index2word, window):
    input_labels = []
    skip_grams = {}
    total_words = np.sum([len(sentence) for sentence in sentences])
    print 'total_words', total_words
    print '# of syms', len(index2word)
    print 'window', window
    for sentence in sentences:
        for pos, word in enumerate(sentence):
            if word not in index2word:
                continue  # it is under min_count
            input_labels.append(index2word.index(word))
            reduced_window = np.random.randint(1, window + 1)
            # print "reduced_window", reduced_window
            # offset = - window + reduced_window
            # for offset in range(-reduced_window, reduced_window+1):

            for offset in range(-window, window + 1):
                key_str = str(offset)
                pos2 = pos + offset
                if offset < -reduced_window or offset > reduced_window or pos2 == pos or \
                   pos2 < 0 or pos2 >= len(sentence):
                    if key_str not in skip_grams.keys():
                        skip_grams[key_str] = [None]
                    else:
                        skip_grams[key_str].append(None)
                    continue
                elif 0 <= pos2 < len(sentence):
                    word2 = sentence[pos2]
                    if word2 not in index2word:
                        word2_ind = None
                    else:
                        word2_ind = index2word.index(word2)
                    if key_str not in skip_grams.keys():
                        skip_grams[key_str] = [word2_ind]
                    else:
                        skip_grams[key_str].append(word2_ind)
                else:
                    assert False, 'ERROR: Case not considered'
    print total_words, len(input_labels)
    # assert total_words == len(input_labels)
    del skip_grams[str(0)]

    input_labels_valid_dict, output_labels_valid_dict = \
        collect_valid_skip_grams(input_labels, skip_grams)
    return input_labels_valid_dict, output_labels_valid_dict


def onehot_label_dict(label_dict, num_labels):
    # num_labels = np.max([ np.max(labels) for labels in label_dict.values() ])
    if isinstance(label_dict, dict):
        onehot_dict = {}
        for key, labels in label_dict.iteritems():
            one_hot_labels = kayak.util.onehot(np.asarray(labels), num_labels=num_labels)
            onehot_dict[key] = one_hot_labels
    else:
        onehot_dict = kayak.util.onehot(np.asarray(label_dict), num_labels=num_labels)
    return onehot_dict


def separate_train_test(inputs_dict, outputs_dict, proportion=0.8):
    if isinstance(inputs_dict, dict):
        train_inputs_dict = {}
        test_inputs_dict = {}
    else:
        print 'input size: %d' % len(inputs_dict)
    train_outputs_dict = {}
    test_outputs_dict = {}
    train_weights_dict = {}
    test_weights_dict = {}

    for key, outputs in outputs_dict.iteritems():
        endi = int(len(outputs) * proportion)
        print endi, len(outputs)
        assert endi < len(outputs)
        train_outputs_dict[key] = outputs[:endi]
        test_outputs_dict[key] = outputs[endi:]

        if isinstance(inputs_dict, dict):
            inputs = inputs_dict[key]
            assert len(outputs) == len(inputs)
            train_inputs_dict[key] = inputs[:endi]
            test_inputs_dict[key] = inputs[endi:]
        train_weights_dict[key] = [item.weight for item in train_inputs_dict[key]]
        test_weights_dict[key] = [item.weight for item in test_inputs_dict[key]]

    if not isinstance(inputs_dict, dict):
        train_inputs_dict = inputs_dict[:endi]
        test_inputs_dict = inputs_dict[endi:]

    if not isinstance(inputs_dict, dict):
        print 'train size: %d' % len(train_inputs_dict)
        print 'test size: %d' % len(test_inputs_dict)

    return train_inputs_dict, train_outputs_dict, train_weights_dict, \
        test_inputs_dict, test_outputs_dict, test_weights_dict


def separate_train_test_arrays(inputs_dict, outputs_dict, proportion=0.8):
    train_inputs_dict = {}
    train_outputs_dict = {}
    test_inputs_dict = {}
    test_outputs_dict = {}

    for key in inputs_dict.keys():
        inputs = inputs_dict[key]
        outputs = outputs_dict[key]

        endi = int(len(inputs) * proportion)
        train_inputs_dict[key] = inputs[:endi, :]
        train_outputs_dict[key] = outputs[:endi, :]
        test_inputs_dict[key] = inputs[endi:, :]
        test_outputs_dict[key] = inputs[endi:, :]

    return train_inputs_dict, train_outputs_dict, test_inputs_dict, test_outputs_dict


def get_train_test_encodings(inputs_dict, outputs_dict,
                             index2word, proportion=0.8,
                             weights_dict=None):
    train_inputs_dict, train_outputs_dict, train_weights, \
        test_inputs_dict, test_outputs_dict, test_weights = \
        separate_train_test(inputs_dict, outputs_dict, proportion)

    num_labels = len(index2word)
    # Use one-hot coding for all
    train_input_labels_dict = onehot_label_dict(train_inputs_dict, num_labels)
    train_output_labels_dict = onehot_label_dict(train_outputs_dict, num_labels)
    test_input_labels_dict = onehot_label_dict(test_inputs_dict, num_labels)
    test_output_labels_dict = onehot_label_dict(test_outputs_dict, num_labels)

    return train_input_labels_dict, train_output_labels_dict, train_weights, \
        test_input_labels_dict, test_output_labels_dict, test_weights


def get_CVs(train_input_labels_dict, train_output_labels_dict, num_fold):
    CVs = {}
    for key, train_output_labels in train_output_labels_dict.iteritems():
        if isinstance(train_input_labels_dict, dict):
            CV = kayak.CrossValidator(num_fold, train_input_labels_dict[key], train_output_labels,
                                      permute=False)
        else:
            CV = kayak.CrossValidator(num_fold, train_input_labels_dict, train_output_labels,
                                      permute=False)
        CVs[key] = CV
    return CVs
