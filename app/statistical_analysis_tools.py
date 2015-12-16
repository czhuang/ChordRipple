
import os

import numpy as np
import scipy.stats as sps
import pylab as plt

import pandas as pd

from collections import defaultdict

from resource import EXPERIMENT_TYPE_STRS


# Single-T, Single-A, Ripple
rankings = {20: '3>2>1', 22:'', 24:'3>2>1', 34: '1>3>2',
 43:'', 355:'2 > 3 > 1', 365:'3>2>1', 444:'3>1>2',
 446:'2 > 3 > 1'}


def get_ranks():
    fpath = os.path.join('csv', 'ranks.csv')
    df = pd.load(fpath)
    return df


def friedmanchisquare_ranks(df):
    print '---rank data'
    print df
    features = []
    for i, condition_id in enumerate(EXPERIMENT_TYPE_STRS):
        features.append(df[condition_id])

    print features
    print len(features)
    statistic, p_value = sps.friedmanchisquare(*features)
    print 'statistic', statistic
    print 'p_value', p_value

def test_ranks():
    ranks = get_ranks()
    friedmanchisquare_ranks(ranks)


def get_data():
    df = pd.load('results.pkl')
    return df


def plot_participants_across_conditions():
    df = get_data()


def friedmanchisquare(df, feature_label):
    print '----', feature_label
    condition_labels = df['condition']

    feature_by_conditions = defaultdict(list)
    for i, cl in enumerate(condition_labels):
        feature_by_conditions[cl].append(df[feature_label][i])

    features = []
    for i, condition_id in enumerate(EXPERIMENT_TYPE_STRS):
        features.append(feature_by_conditions[condition_id])

    print features
    print len(features)
    statistic, p_value = sps.friedmanchisquare(*features)
    print 'statistic', statistic
    print 'p_value', p_value


def subset_friedmanchisquare(df, feature_label, qualifying_feature_label):
    print '----', feature_label
    condition_labels = df['condition']

    feature_by_conditions = defaultdict(list)
    qualifying_feature_by_conditions = defaultdict(list)
    for i, cl in enumerate(condition_labels):
        feature_by_conditions[cl].append(df[feature_label][i])
        qualifying_feature_by_conditions[cl].append(df[qualifying_feature_label][i])

    features = []
    num_participants = len(feature_by_conditions.values()[0])
    qualifying_inds = range(num_participants)
    for i, condition_id in enumerate(EXPERIMENT_TYPE_STRS):
        local_features = feature_by_conditions[condition_id]
        local_qualifying_features = qualifying_feature_by_conditions[condition_id]
        for j, feat in enumerate(local_features):
            if local_qualifying_features[j] < 4 and j in qualifying_inds:
                qualifying_inds.remove(j)
    print 'qualifying_inds', qualifying_inds
    # qualifying_inds [0, 1, 3, 4, 5] < 4
    # qualifying_inds [0, 1, 3, 4, 5, 7, 8] < 3.5

    features = []
    for i, condition_id in enumerate(EXPERIMENT_TYPE_STRS):
        local_features = [feature_by_conditions[condition_id][i] for i in qualifying_inds]
        features.append(local_features)

    print features
    print len(features)
    statistic, p_value = sps.friedmanchisquare(*features)
    print '# of qualifying participants:', len(qualifying_inds)
    print 'statistic', statistic
    print 'p_value', p_value


def test_qualify_novelty_with_goodness_rating():
    df = get_data()
    subset_friedmanchisquare(df, 'best_seq-log_mean_unigram_inverse_freq',
                             'best_seq-rating')


def run_tests():
    df = get_data()
    feature_labels = ['best_seq-rating', 'best_seq-loglikelihood',
                      'best_seq-log_mean_unigram_inverse_freq']
    for feature_label in feature_labels:
        friedmanchisquare(df, feature_label)


if __name__ == '__main__':
    # run_tests()
    # test_qualify_novelty_with_goodness_rating()

    test_ranks()