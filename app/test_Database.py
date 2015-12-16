
import cPickle as pickle

import numpy as np

from Database import Database
from Database import DOC_TYPE_OBJECTS

from make_object_tools import make_dummy_QueryObject


def test_class_to_json():
    db = Database()
    query_obj = make_dummy_QueryObject()
    print db.format(query_obj)


def test_index():
    db = Database()
    query_obj = make_dummy_QueryObject()
    doc_type = DOC_TYPE_OBJECTS.keys()[0]
    print 'doc_type', doc_type
    db.index(doc_type, query_obj)
    db.retrieve_all()


def test_mapping():
    db = Database()


def test_access_field():
    # needs mapping in order to do this
    db = Database()
    query_obj = make_dummy_QueryObject()
    doc_type = DOC_TYPE_OBJECTS.keys()[0]
    print 'doc_type', doc_type
    db.index(doc_type, query_obj)

    query_obj.play = True
    db.index(doc_type, query_obj)

    db.count_num_docs()

    # TODO: hmm... without mapping can do this too
    db.search_match_term({"play": True})
    db.search_match_term({"seq_str": 'Cm'})


def test_index_model():
    from retrieve_model_tools import retrieve_NGram, retrieve_SkipGramNN
    ngram = retrieve_NGram()
    nn = retrieve_SkipGramNN()

    assert ngram.syms == nn.syms

    db = Database()
    db.index_model(ngram, 'ngram')
    db.index_model(nn, 'nn')

    query_obj = make_dummy_QueryObject()
    db.index("user_actions", query_obj)

    db.search_match_term(dict(docType="models"))
    # db.retrieve_all()


def test_index_retrieve_user_actions():
    db = Database(which='pilot')
    query_obj = make_dummy_QueryObject()
    db.index_user_action(query_obj, 'Ripple')

    hits = db.search_match_term(dict(docType="user_actions"))
    for hit in hits:
        print 'experiment_type_label' in hit["_source"].keys()

def test_retreive_pilot_user_actions():
    test_retrieve_user_actions('pilot')


def test_retreive_test_user_actions():
    test_retrieve_user_actions('test')


def test_retrieve_user_actions(db_index):
    db = Database(db_index)
    print '=== check total counts ==='
    db.count_num_docs()
    hits = db.search_match_term(dict(docType="user_actions"))
    print '# of hits:', len(hits)
    times = []
    for hit in hits:
        source = hit["_source"]
        times.append(source["time"])
    print 'times', times
    sorted_indices = np.argsort(times)
    print 'sorted_indices', sorted_indices

    print '---sorted list---'
    userId_dict = {}
    for idx in sorted_indices:
        source = hits[idx]["_source"]

        userId = source["userId"]
        if userId not in userId_dict:
            userId_dict[userId] = len(userId_dict.keys())

        if "data" in source:
            print source["data"]

        print source["actionKind"], db.format_datetime(source["time"]) , "user_%d" % userId_dict[userId],
        # if source["actionKind"] == 'inputFocus':
        print source["activeIdx"], source["actionAuthor"], source["author"]
        if source['actionKind'] == 'rating':
            if 'ratingQuestion' in source:
                print source['ratingQuestion']
            else:
                print source["seqStr"]
        else:
            print source["seq"], source["seqStr"]
        if 'originalText' in source:
            print 'originalText', source['originalText']
        if 'rating' in source:
            print 'rating', source['rating']
        if "experiment_type_label" in source:
            print source["experiment_type_label"]
        else:
            print



    fname = '%s_logs.pkl' % db_index
    with open(fname, 'wb') as p:
        pickle.dump(hits, p)


def test_pilot_analysis(db_index):
    # db = Database('pilot', True)
    db = Database(db_index, False)
    db.retrieve_hits_from_pickle(db_index)
    db.preprocess_hits()

    # print '--- serious ---'
    # db.unique_chords(db.serious_user_keys)
    #
    # print '--- casual ---'
    # db.unique_chords(db.casual_user_keys)
    #
    # print '--- serious ---'
    # db.summarize_chord_changes_by_step(db.serious_user_keys)
    #
    print '--- casual ---'
    # db.summarize_chord_changes_by_step(db.casual_user_keys)

    # print '--- novelty serious ---'
    # db.summarize_novelty_theta_by_step(db.serious_user_keys)
    # print '--- novelty casual ---'
    # db.summarize_novelty_theta_by_step(db.casual_user_keys)

    # print '--- time series serious ---'
    # db.theta_as_time_series(db.serious_user_keys)
    # db.theta_as_time_series(db.casual_user_keys)

    print '--- last seq novelty casual ---'
    # db.last_seq_novelty(db.casual_user_keys)

    # db.last_seq_novelty(db.serious_user_keys)
    #
    # db.last_seq_novelty(db.complete_user_keys)

    # db.process_against_final(db.complete_user_keys)

    # db.collect_use_actions()
    # db.collect_likelihood_versus_is_ripple()
    # db.collect_use_dependencies()

    # db.print_complete_hits()

# def test_get_all_use():


if __name__ == '__main__':
    # test_class_to_json()
    # test_index()
    # test_mapping()
    # test_access_field()

    # test_index_model()
    # test_retrieve_model()

    # test_retrieve_user_actions("study-iui")
    # test_index_retrieve_user_actions()

    # test_retreive_pilot_user_actions()
    # test_retreive_test_user_actions()

    test_pilot_analysis("study-iui")

