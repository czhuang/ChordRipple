
"""
This module contains the database classes for setting up an Elastic
Search index, storing and retrieving user-interaction logs to and from
that index, and also for some simple analysis and plotting of the logs.
"""

import os
import copy

import csv

import cPickle as pickle
from collections import defaultdict

import numpy as np
import pandas as pd
DEPLOY = True
if not DEPLOY:
    import seaborn as sns
    # import pandas as pd
    import matplotlib.pyplot as plt


from datetime import datetime
import time
from collections import OrderedDict
import hashlib
import json
import elasticsearch as es
import jsonpickle

from music21 import harmony
from music21_chord_tools import preprocess_letters_before_sym2chord

from QueryObject import QueryObject
from make_object_tools import make_dummy_QueryObject
from retrieve_model_tools import retrieve_NGram, retrieve_SkipGramNN

from utility_tools import print_vector, print_array


EXPERIMENT_TYPE_STRS = ['Single-T', 'Single-A', 'Ripple']


# DB_DOC_TYPES = ['models', 'user_actions', 'user_background_surveys',
#                 'user_feedback']

DOC_TYPE_OBJECTS = OrderedDict(user_actions=make_dummy_QueryObject,
                               models=None)

STR_STORED = dict(type="string", store=True)
FLOAT_STORED = dict(type="float", store=True)
INT_STORED = dict(type="integer", store=True)
BOOLEAN_STORED = dict(type="boolean", store=True)

DELETE_INDEX_IF_ALREADY_EXIST = False
SEARCH_RESULT_SIZE = 100000

VALID_SYMS = [u'C', u'F', u'Bb', u'Gm', u'Cm', u'G', u'Ab', u'Gb', u'Am/C', u'Ab7', u'Am', u'Fmaj7/C', u'Ab/C', u'Cmaj7', u'Am7', u'B', u'F#o7/Eb', u'Dm7', u'G11', u'E/B', u'Eb/Bb', u'E7/G#', u'FM9', u'Em7', u'Am/E', u'D7', u'Dm', u'Bb/F', u'Gsus', u'Gsus2', u'Gsus4', u'F#o7/C', u'FmM7', u'F/C', u'F#dim', u'Bbb7', u'Am9', u'Gm7', u'CmM7/B', u'Dm7/A', u'F#m7', u'Bm7', u'Bbm7', u'Bb7', u'F/A', u'C#dim', u'Eb', u'a', u'Fmaj7/A', u'Fm', u'A', u'C7', u'F7', u'c', u'g', u'E', u'Em', u'D', u'b', u'e', u'd', u'Ddim', u'F/a', u'C/G', u'G/B', u'Fmaj', u'Fminmaj7', u'Faug', u'Faug7', u'C/E', u'G/D', u'C#', u'C#/D', u'C#/G', u'A#', u'Em/B', u'Adim', u'Fmaj7', u'G7', u'E7', u'Bdim', u'Fdim', u'F#', u'Em9', u'Csus4', u'Cm7']

NOT_VALID_SYMS = [u'm7', u'GH', u'r', u'CD', u'Fminmaj', u'F:minmaj7', u'F:minmaj7', u'F:minmaj7', u'Fa', u'C#/', u'C#/', u'Fmaj7/', u'Fd', u'Fdi', u'Emsus2', u'Emsus2', u'Emsus2', u'Emsus2', u'Emsus2', u'Emsus2', u'Emsus2', u'Emsus2', u'Emsus2', u'C7sus', u'C7sus4', u'C7sus4', u'C7sus4', u'C7sus4', u'C7sus4', u'C7sus4', u'm', u'Am/']

CHORD_LEN = 8

PLOT_DIR = 'plots'
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

UNKNOWN_EMPTY_SYMBOL = 'Unknown'

# These are for entries that got generated when I was testing the system
# before I started adding "test" to the comments
USER_ID_IGNORE_LIST = ['20']


def make_hash(x):
    x = jsonpickle.encode(x)
    return hashlib.md5(json.dumps(x)).hexdigest()


def delta_time(dt, dt1):
    return (dt1 - dt).total_seconds()


class Keys(list):
    def __init__(self, keys, name):
        list.__init__(self)
        self.extend(keys)
        self.name = name


class DatabasePlacebo(object):
    def __init__(self):
        pass
        # print 'WARNING: Using Database Placebo, that means we are not logging anything'
        # response = raw_input('Are you sure? [Y/n] ')
        # if response.strip() == 'n':
        #     assert False, 'Needed logging but database not setup'

    def connect(self):
        pass

    def index_model(self, obj, name):
        pass

    def index_user_action(self, query_obj, experiment_type_label,
                          suggestion_objs=None,
                          attrs=None):
        pass


class Database(object):
    """
    This class sets up an Elastic Search index, to store and retrieve
    user-interaction logs to and from that index, and also for some
    simple analysis and plotting of the logs.
    """
    

    def __init__(self, which, connect=True):
        """
        Instantiates a database object which connects to an Elastic Search index.
        It also creates a unique user ID for the session, and retrieves the
        corresponding models that are needed for analysis.

        :param which: str
            String that identifies which database to connected to

        :param connect: boolean, optional
            Boolean for if we want to connect now or not
        :return: Database
            An instance of the Database class

        """

        if which == 'test':
            self.index_id = 'chordripple-v2-dev-test'
        elif which == 'pilot':
            self.index_id = 'chordripple-v2-pilot'
        elif which == 'study-iui':
            self.index_id = 'chordripple-v2-study-iui'
        else:
            assert False, 'ERROR: case not implemented'

        self.userID = make_hash(datetime.utcnow())
        print '...userID', self.userID
        if connect:
            self.db = self.connect()
        self.doc_type_objects = DOC_TYPE_OBJECTS
        # self.make_mappings()
        # self.make_mappings_model()
        self.valid_syms = {}
        self.not_valid_syms = NOT_VALID_SYMS

        self.ngram = retrieve_NGram()
        # self.nn = retrieve_SkipGramNN()

        self.condition_ids = EXPERIMENT_TYPE_STRS
        self.results = pd.DataFrame()
        # self.res = {}

    def connect(self):
        """
        Creates an Elasticsearch instance, establishes a connection with
        the ES database, and creates an index in the database if it didn't
        exist already.

        :return: Elasticsearch instance
        """

        host = '54.175.10.3'
        port = 9200
        
        print '...Connecting to database at host: %s, port:%d' % (host, port)
        login = dict(host=host, port=port)
        db = es.Elasticsearch(hosts=[login])
        if not db.ping():
            raise es.ConnectionError(
                "Failed to connect to {host}:{port}".format(**login))
        else:
            print 'Successfully connected to database...'

        # Can directly create an index if not already exist
        # if DELETE_INDEX_IF_ALREADY_EXIST:
        #     if db.indices.exists(index=self.index_id):
        #         print "Clearing index %s at %s" % (self.index_id, host)
        #         db.indices.delete(index=self.index_id)
        #
        #     result = db.indices.create(index=self.index_id)
        #     if not result.get("acknowledged", False):
        #         raise es.RequestError("Could not create index: '{0}'"
        #                               "".format(self.index_id))
        if not db.indices.exists(index=self.index_id):
            result = db.indices.create(index=self.index_id)
            if not result.get("acknowledged", False):
                raise es.RequestError("Could not create index: '{0}'"
                                      "".format(self.index_id))
        return db

    def prepare_index(self, doc_type, attrs):
        store_dict = dict(userId=self.userID, docType=doc_type)
        store_dict['datatime'] = datetime.utcnow()
        # so that can sort? but want to first group by user
        store_dict['time'] = time.time()
        # print '...prepare_index', attrs
        if attrs is not None:
            assert isinstance(attrs, dict), 'ERROR: attrs need to be a dictionary'
            for key, obj in attrs.iteritems():
                # not using original key
                if isinstance(obj, str) or isinstance(obj, int) or isinstance(obj, float):
                    store_dict[key] = obj
                else:
                    store_dict.update(self.format(obj, store_attrs=False))
        # print 'store_dict', store_dict
        return store_dict

    def format(self, obj, store_attrs=True):
        store_dict = {}
        # TODO: resolve hack parent ref
        if hasattr(obj, 'parent'):
            parent = obj.parent
            obj.parent = None
            obj_json = jsonpickle.encode(obj)
        else:
            obj_json = jsonpickle.encode(obj)

        obj_type = type(obj).__name__
        store_dict[obj_type+"_json"] = obj_json

        if store_attrs:
            # TODO:  add a function in each object to serialize
            # so that can add side information
            for attr in obj.__dict__.keys():
                store_dict[attr] = getattr(obj, attr)

        # for key, val in store_dict.iteritems():
        #     print key, ':', val
        # return json.dumps(store_dict)

        # TODO: resolve hack, restore parent ref
        if hasattr(obj, 'parent'):
            obj.parent = parent

        return store_dict

    def make_mapping_from_obj(self, obj, doc_type):
        properties = dict()
        store_dict = self.format(obj, doc_type)
        for key, val in store_dict.iteritems():
            if isinstance(val, bool):
                properties[key] = BOOLEAN_STORED
            elif isinstance(val, int):
                properties[key] = INT_STORED
            elif isinstance(val, float):
                properties[key] = FLOAT_STORED
            elif isinstance(val, str):
                properties[key] = STR_STORED
            else:
                properties[key] = STR_STORED

        print 'properties'
        for key, val in properties.iteritems():
            print key, val
        mapping = dict(properties=properties)
        return mapping

    def make_mappings(self):
        # TODO: which one of the class attributes need to be searched?
        for doc_type, make_dummy_obj_func in self.doc_type_objects.iteritems():
            if make_dummy_obj_func is None:
                continue
            print 'doc_type', doc_type
            obj = make_dummy_obj_func()
            print 'obj', obj
            mapping = self.make_mapping_from_obj(obj, doc_type)
            result = self.db.indices.put_mapping(
                index=self.index_id, doc_type=doc_type,
                body={doc_type: mapping})
            print 'make mappings:', result

    def make_mappings_model(self):
        doc_type = 'models'
        mapping = {"properties": {"model": dict(type="string",
                                                index="not_analyzed")}}

        result = self.db.indices.put_mapping(
                index=self.index_id, doc_type=doc_type,
                body={doc_type: mapping})
        print 'make mappings:', result

    def index_model(self, obj, name):
        doc_type = 'models'
        assert doc_type in self.doc_type_objects.keys()
        # body = self.format(obj, store_attrs=False,
        #                    attrs={"model_type": name})
        body = dict(modelHash=make_hash(obj), modelType=name, docType=doc_type)
        res = self.db.index(index=self.index_id,
                            doc_type=doc_type,
                            body=body)
        print '...ES, index', res['created'], doc_type#, obj

    def index_user_action(self, query_obj, experiment_type_label,
                          suggestion_objs=None,
                          attrs=None):
        # print '...db: type(query_obj)', type(query_obj)
        doc_type = 'user_actions'
        assert doc_type in self.doc_type_objects.keys()
        expr_dict = dict(experiment_type_label=experiment_type_label)
        if attrs is not None:
            expr_dict.update(attrs)
        body = self.prepare_index(doc_type, expr_dict)
        query_body = self.format(query_obj)
        body.update(query_body)

        # print '...db: finished with query, moving onto suggestions'
        if suggestion_objs is not None:
            for suggestion_obj in suggestion_objs:
                # print '...db: type(suggestion_obj)', type(suggestion_obj)
                suggestion_body = self.format(suggestion_obj, False)
                body.update(suggestion_body)

        res = self.db.index(index=self.index_id,
                            doc_type=doc_type, body=body)
        print '...ES, index', res['created'], doc_type

    # def get(self, doc_type):
    #     res = self.db.get(index=self.index_id, doc_type=doc_type)
    #     print '...ES, get', res['_source'], doc_type

    def retrieve_all(self, doc_type=None):
        assert doc_type is None or doc_type in self.doc_type_objects.keys()
        self.db.indices.refresh(index=self.index_id)
        query = {"match_all": {}}
        if doc_type is None:
            res = self.db.search(index=self.index_id,
                                 body={"query": query})
        else:
            res = self.db.search(index=self.index_id, doc_type=doc_type,
                                 body={"query": query})

        print '...ES', "Got %d Hits:" % res['hits']['total']
        # for hit in res['hits']['hits']:
        #     print hit["_source"].keys()
        #     print '...ES', "%(timestamp)s %(author)s: %(seq_str)s" % hit["_source"]
        #     print '...ES', hit["_source"]

    def count_num_docs(self):
        self.db.indices.refresh(index=self.index_id)
        res = self.db.count(index=self.index_id, body={"query": {"match_all": {}}})
        print '...ES', "# of docs: %d" % res['count']

    def search_match_term(self, match_dict):
        # TODO: note that a lof of them are strings
        self.db.indices.refresh(index=self.index_id)
        # query = {"term": {"play": None}}

        query = {"term": match_dict}

        res = self.db.search(index=self.index_id, size=SEARCH_RESULT_SIZE,
                             body={"query": query})
        # print '--- hits ---'
        # print res['hits']
        print '...ES', "Got %d Hits:" % res['hits']['total']
        # for hit in res['hits']['hits']:
        #     print "hit.keys()", hit.keys()
        #     print hit["_source"].keys()
        #     # print '...ES', "%(timestamp)s %(author)s: %(seq_str)s" % hit["_source"]
        #     # print '...ES', hit["_source"]["play"]
        #     pass
        return res['hits']['hits']

    def retrieve_hits_from_pickle(self, db_index):
        fname = '%s_logs.pkl' % db_index
        with open(fname, 'rb') as p:
            self.hits = pickle.load(p)
        print '# of hits retrieved:', len(self.hits)

    def preprocess_hits(self):
        self.sort_indices()
        self.separate_user_logs()

        self.extract_results_from_controlled_studies()


        # self.filter_complete_user_entries()
        # self.extract_entries_with_only_chord_changes()
        # self.filter_last_complete_seqs()
        # self.filter_first_complete_seqs()
        # # self.collect_chords_used_collaspsed()
        # self.collect_chord_changes_by_step()
        # self.time_spent()
        # self._unique_chords()
        # self._last_seq_novelty()
        # self.summarize_novelty_by_step()
        # self._top_novelty_in_changes()
        # self._novelty_std_in_changes()

        # print '--- valid_syms ---'
        # print self.valid_syms
        #
        # print '--- not_valid_syms ---'
        # print self.not_valid_syms

    def print_complete_hits(self):
        for key in self.complete_user_keys:
            print '----- User %d' % key
            for seq in self.chord_change_entries[key]:
                self.print_seq(seq)

    def sort_indices(self):
        assert hasattr(self, 'hits')
        times = []
        for hit in self.hits:
            source = hit["_source"]
            times.append(source["time"])
        print 'times', times
        sorted_indices = np.argsort(times)
        print 'sorted_indices', sorted_indices
        self.sorted_indices = sorted_indices

    def get_hit_source(self, idx):
        return self.hits[idx]["_source"]

    def separate_user_logs(self):
        userId_dict = {}
        user_entries = {}
        for idx in self.sorted_indices:
            source = self.get_hit_source(idx)
            userId = source["userId"]

            if userId not in userId_dict:
                userId_dict[userId] = len(userId_dict.keys())

                # this was a filtering hack
                # if len(userId_dict.keys()) > 10:
                user_num = userId_dict[userId]
                user_entries[user_num] = [source]
            else:
                user_num = userId_dict[userId]
                # if len(userId_dict.keys()) > 10:
                user_entries[user_num].append(source)
        self.userId_dict = userId_dict
        self.user_entries = user_entries
        for key, entries in user_entries.iteritems():
            print key, self.format_datetime(entries[0]['time']), \
                entries[-1]['experiment_type_label'], len(entries)

    def check_all_user_seq_complete(self):
        for user_hash, user_id in self.user_id_map.iteritems():
            self.check_user_seq_complete(user_id)

    def user_id_to_user_hash(self, user_id_requested):
        for user_hash, user_id in self.user_id_map.iteritems():
            if user_id_requested == user_id:
                return user_hash
        assert False, 'ERROR: could not find user_id'

    def is_complete_seq(self, seq):
        # emptychord = '' in seq
        stripped_seq = [sym.strip() for sym in seq]
        emptychord = UNKNOWN_EMPTY_SYMBOL in seq
        # print 'stripped_seq', stripped_seq
        if not emptychord and len(seq) == CHORD_LEN:
            return True
        else:
            return False

    def is_test(self, entries):
        return False
        # for entry in entries:
        #     if entry['actionKind'] == "comments" and \
        #                     entry['seqStr'].strip == "test":
        #         return True
        # return False

    def last_complete_seq(self, entries):
        # assumes that it is sorted
        # print '---last_complete_seq'
        # for entry in entries:
        #     print entry["seq_formatted"]
        for i in range(len(entries)-1, -1, -1):
            if self.is_complete_seq(entries[i]["seq_formatted"]):
                return entries[i]["seq_formatted"]
        return None

    def first_complete_seq(self, entries):
        # assumes that it is sorted
        reached = None
        for i in range(len(entries)):
            if self.is_complete_seq(entries[i]):
                reached = i
                # entries[i]["reached_first_complete"] = i
                break
        # if reached is None:
        #     print '\n------ # of entries', len(entries)
        #     print entries[-1]
        # else:
        #     print '\n------ reached at %d -----' % reached
        #     for j, seq in enumerate(entries):
        #         if j == i:
        #             print j, seq
        #         else:
        #             print seq
        if reached is not None:
            return entries[reached], reached
        else:
            return reached, len(entries)

    def filter_entries_to_only_chord_seqs(self, entries):
        stripped_entries = []
        for entry in entries:
            if not entry['actionKind'] == 'comments':
                seq = entry['seq']
                formatted_seq = []
                for sym in seq:
                    formatted_sym, success = self.validate_chord_symbol(sym)
                    if success:
                        formatted_seq.append(formatted_sym)
                    else:
                        formatted_seq.append(UNKNOWN_EMPTY_SYMBOL)
                entry['seq_formatted'] = formatted_seq
                stripped_entries.append(entry)
        return stripped_entries

    def extract_entries_with_only_chord_changes(self):
        chord_change_entries = defaultdict(list)
        for key in self.complete_user_keys:
            entries = self.complete_entries[key]
            seq_p = entries[0]["seq_formatted"]
            chord_change_entries[key].append(seq_p)
            for entry in entries[1:]:
                seq = entry["seq_formatted"]
                if seq_p != seq:
                    chord_change_entries[key].append(seq)
                seq_p = seq
        self.chord_change_entries = chord_change_entries

    def format_datetime(self, raw_time):
        # raw_time += 60*60*5
        return datetime.fromtimestamp(raw_time).strftime('%Y-%m-%d %H:%M:%S')

    def collect_conditions(self, entries):
        conditions = []
        entries_by_condition = defaultdict(list)
        for entry in entries:
            expr_label = entry['experiment_type_label']
            if len(conditions) == 0:
                conditions.append(expr_label)
            elif expr_label != conditions[-1]:
                conditions.append(expr_label)
                # hack to change the first Ripple condition to tutorial
                if expr_label == 'Ripple':
                    assert conditions[0] == 'Ripple'
                    entries_by_condition[conditions[0]+'-tutorial'] = entries_by_condition[conditions[0]]
                    del entries_by_condition[conditions[0]]
            entries_by_condition[expr_label].append(entry)

        for key, entries in entries_by_condition.iteritems():
            print key, len(entries)
        assert self.check_all_conditions_present(entries_by_condition.keys()),\
            'condition name not found'

        return entries_by_condition

    def check_all_conditions_present(self, keys):
        for name in self.condition_ids:
            if name not in keys:
                return False
        return True


    def filter_complete_user_entries_controlled(self, add=True):
        print '--- filter_complete_user_entries_controlled'
        complete_entries = defaultdict(list)
        for user_id, entries in self.user_entries.iteritems():
            conditions = []
            for entry in entries:
                expr_label = entry['experiment_type_label']
                if len(conditions) == 0:
                    conditions.append(expr_label)
                elif expr_label != conditions[-1]:
                    conditions.append(expr_label)
            # print user_id, self.format_datetime(entry['time']), \
            #         conditions, len(entries)
            if len(conditions) >= 3 and self.check_all_conditions_present(conditions):
                complete_entries[user_id] = entries
                print user_id, self.format_datetime(entry['time']), \
                    conditions
        print '# of :', len(complete_entries)
        complete_entries_by_conditions = {}
        sorted_keys = np.sort(complete_entries.keys())
        for key in sorted_keys:
            entries = complete_entries[key]
            print '----', self.format_datetime(entries[0]['time']), ',', key
            entries_by_conditions = self.collect_conditions(entries)
            # all entries need to be
            all_enough_entries = True
            for condition_id in self.condition_ids:
                if len(entries_by_conditions[condition_id]) < 50:
                    print '--', condition_id, 'not enough entries'
                    all_enough_entries = False
                    break
            if all_enough_entries:
                complete_entries_by_conditions[key] = entries_by_conditions

        # exception_user_ids = [9], Single-A, 30, but don't have ratings

        print '\n\n---------------'
        self.complete_entries_by_conditions = complete_entries_by_conditions
        print '# of complete entries:', len(self.complete_entries_by_conditions)

        # somehow no Single-T saved sequences for user_id 37...
        INCOMPLETE = [37]
        for user_id in INCOMPLETE:
            del self.complete_entries_by_conditions[user_id]

        self.complete_entries_by_conditions_keys = list(np.sort(self.complete_entries_by_conditions.keys()))
        for user_id in self.complete_entries_by_conditions_keys:
            entries = self.complete_entries_by_conditions[user_id]
            print '----', self.format_datetime(entries.values()[0][0]['time']), ',', user_id
            for key, condition_entries in entries.iteritems():
                print key, len(condition_entries)
            # Don't remove this because contains ordering info for example
            # if 'Ripple-tutorial' in entries.keys():
            #     del entries['Ripple-tutorial']

        if add:
            self.results['user_id'] = self.complete_entries_by_conditions_keys


    # ==============
    #  the main for extracting results
    # ==============
    def extract_results_from_controlled_studies(self):
        # creates data frame with user_id column
        self.filter_complete_user_entries_controlled(add=False)
        self.collect_experiment_ordering(add=False)

        # udpates the self.result dataframe
        self.collect_best_rating_seq_by_condition(add=False)
        self.collect_best_seq_likelihood()
        # self.print_best_seq_rating_and_likelihood()

        self.collect_best_seq_mean_novelty()

        self.compute_goodness_cross_novelty()

        # self.collect_rating_questions()
        self.collect_comments()

        self.stack_ratings_scores()
        self.print_results()
        self.format_rankings()
        self.collect_uses()

        fpath = os.path.join('csv', 'ordering.csv')
        self.results.to_csv(fpath)
        self.results.to_pickle('results.pkl')

    def stack_ratings_scores(self):
        QUESTIONS = False
        columns = ['user_id', 'condition', 'order', 'best_seq-rating',
                     'best_seq-loglikelihood',
                     'best_seq-log_mean_unigram_inverse_freq',
                     'best_seq-novelty_scaled',
                     'best_seq-goodness_scaled',
                     'best_seq-novelty_goodness_prod',
                     'best_seq-novelty_goodness_average']
        if QUESTIONS:
            columns.extend(copy.copy(self.question_keywords))
        columns.append('best_seq')

        self.results = pd.DataFrame(columns=columns)
        row_i = 0
        for user_id in self.complete_entries_by_conditions:
            for i, condition_id in enumerate(self.condition_ids):
                ordering = self.ordering_by_user[user_id]
                rating = self.best_seq_ratings_by_condition[user_id][condition_id]
                ll = self.likelihoods_by_condition[user_id][condition_id]
                # log_mean_unigram_inverse_freq = self.unigram_inverse_freq[user_id][condition_id]['log_mean']
                log_mean_unigram_inverse_freq = self.unigram_inverse_freq[user_id][condition_id]['mean']

                novelty_goodness_prod = self.novelty_goodness_by_condition[user_id][condition_id]['novelty_goodness_prod']
                novelty_goodness_average = self.novelty_goodness_by_condition[user_id][condition_id]['novelty_goodness_average']

                novelty_scaled = self.novelty_goodness_by_condition[user_id][condition_id]['novelty_scaled']
                goodness_scaled = self.novelty_goodness_by_condition[user_id][condition_id]['goodness_scaled']

                best_seq = self.best_seq_by_condition[user_id][condition_id][1]
                res = [user_id, condition_id,
                       ordering.index(i), rating,
                       ll, log_mean_unigram_inverse_freq,
                       novelty_scaled, goodness_scaled,
                       novelty_goodness_prod,
                       novelty_goodness_average]

                if QUESTIONS:
                    ratings = []
                    for keyword in self.question_keywords:
                        rating = self.questions[user_id][condition_id][keyword]
                        ratings.append(rating)

                    res.extend(ratings)
                res.append(best_seq)

                self.results.loc[row_i] = res
                row_i += 1
        print self.results

    def contain_keywords(self, keywords, question):
        if not isinstance(keywords, list):
            if keywords in question:
                return True
            else:
                return False
        for word in keywords:
            if word not in question:
                return False
        return True

    def questions_completed(self, nested_dict, keywords):
        for condition_id in self.condtion_ids:
            if condition_id not in nested_dict:
                return False
            else:
                for word in keywords:
                    if word not in nested_dict[condition_id]:
                        return False
        return True

    def collect_rating_questions(self):
        self.question_keywords = ['overall', 'songwriting']
        self.questions = {}

        for user_id in self.complete_entries_by_conditions_keys:
            user_ratings = {}
            for condition_id in self.condition_ids:
                entries = self.complete_entries_by_conditions[user_id][condition_id]
                print '-----'
                for entry in entries:
                    print entry['actionKind']
                for keyword in self.question_keywords:
                    for entry in entries:
                        if 'ratingQuestion' in entry and keyword in entry['ratingQuestion']:
                            print user_id, entry['rating'], keyword, user_id, condition_id, entry['ratingQuestion']
                            user_ratings[keyword] = entry['rating']
            # need to check for completeness
            # some ratings are not re-entered so then they are not saved
            self.questions[user_id] = user_ratings

    def collect_comments(self):
        self.comments = {}
        for user_id in self.complete_entries_by_conditions_keys:
            self.comments[user_id] = {}
            ordering = self.ordering_by_user[user_id]
            ordering = [int(idx) for idx in ordering]
            print '--------', ordering
            for idx in ordering:
                print self.condition_ids[idx],
            print
            for idx in ordering:
                condition_id = self.condition_ids[idx]
                entries = self.complete_entries_by_conditions[user_id][condition_id]
                for entry in entries:
                    if entry['actionKind'] == 'comments':
                        self.comments[user_id][condition_id] = entry['seqStr']
                        print user_id, condition_id
                        print '\t', entry['seqStr']

    def use_loglikelihood_diff(self, entry):
        print 'use_loglikelihood_diff'
        assert entry['actionKind'] == 'use' and entry['author'] == 'machine'
        changed_indices = self.get_changed_indices(entry)
        seq = entry['seq']
        print 'changed_indices', changed_indices
        if len(changed_indices) == 1:
            idx = changed_indices[0]
            sym = seq[idx]
            # only want entries where there can be complete ripples
            if idx > 0 and idx < len(seq) - 2:
                # if single chord substitute into original
                original_subseq = seq[idx-1:idx+2]
                print 'original_subseq', original_subseq
                subseq_ll = self.ngram.log_likelihood(original_subseq)
                attr_name = "SuggestionList" + "_json"
                # want to retrieve how ripple subseq would be like
                if attr_name in entry:
                    obj_json = entry[attr_name]
                    suggestions = jsonpickle.decode(obj_json)
                    ripple_ll = None
                    ripple_subseq = None
                    print 'suggestions.items', len(suggestions.items), suggestions.items
                    if len(suggestions.items) == 0:
                        return 'ERROR'
                    print 'sym', sym, ''
                    for item in suggestions.items:
                        print item.seq
                        print item.inds
                        print item.kind
                        if sym in item.seq and item.inds == range(idx-1, idx+2):
                            print 'got it'
                            ripple_subseq = item.seq[idx-1:idx+2]
                            ripple_ll = self.ngram.log_likelihood(ripple_subseq)

                    # assert ripple_ll is not None
                    if ripple_ll is None:
                        return 'ERROR'
            else:
                return None
        elif len(changed_indices) == 3:
            # how ripple subseq is like
            ripple_subseq = seq[changed_indices[0]:changed_indices[-1]+1]
            ripple_ll = self.ngram.log_likelihood(ripple_subseq)
            sym = seq[changed_indices[1]]
            original_seq = entry['originalText']
            # if sub into original seq
            original_subseq = [original_seq[changed_indices[0]]] + [sym] +\
                [original_seq[changed_indices[2]]]
            subseq_ll = self.ngram.log_likelihood(original_subseq)
        else:
            return None

        print 'original_subseq', original_subseq
        print 'ripple_subseq', ripple_subseq
        print 'subseq_ll', subseq_ll
        print 'ripple_ll', ripple_ll
        diff = np.exp(ripple_ll) - np.exp(subseq_ll)
        # log_diff = np.log(diff)
        # return ripple_ll, subseq_ll, log_diff
        return ripple_ll, subseq_ll, diff

    def collect_uses(self):
        self.uses = {}
        error_count = 0
        error_indices = []
        for user_id in self.complete_entries_by_conditions_keys:
            entries = self.complete_entries_by_conditions[user_id]['Ripple']
            results = []
            for entry in entries:
                if entry['actionKind'] == 'use' and entry['author'] == 'machine':
                    print '---- use ----'
                    res = {}
                    changed_indices = self.get_changed_indices(entry)
                    # diff = self.use_loglikelihood_diff(entry)
                    # if diff is None:
                    #     continue
                    # if diff == 'ERROR':
                    #     error_count += 1
                    #     error_indices.append(changed_indices)
                    #     continue
                    # for i, label in enumerate(['ripple_ll', 'original_ll', 'diff']):
                    #     res[label] = diff[i]
                    seq = entry['seq']
                    original_seq = entry['originalText']
                    print seq
                    print original_seq
                    # if side ripples, then skip
                    if len(changed_indices) == 2:
                        continue
                    if self.is_ripple(entry):
                        res['is_ripple'] = 1
                        assert len(changed_indices) == 3
                        sym = seq[changed_indices[1]]
                    else:
                        res['is_ripple'] = 0
                        assert len(changed_indices) == 1
                        sym = seq[changed_indices[0]]
                    res['novelty'] = self.ngram.unigram_inverse_freq(sym)
                    if res['novelty'] is not None:
                        results.append(res)
            self.uses[user_id] = results

        results = []
        for user_id in self.complete_entries_by_conditions_keys:
            results.extend(self.uses[user_id])

        df = pd.DataFrame(results)
        # print 'error_count', error_count
        # for indices in error_indices:
        #     print indices
        #
        # diffs = df['diff']
        # min = np.min(diffs)
        # # max = np.max(diffs)
        # # range_ = max - min
        # diffs = diffs - min
        # diffs = np.log(diffs)
        # df['diff'] = diffs

        is_ripple = df['is_ripple']
        print '# of use, # of ripple use, percentage'
        print len(is_ripple), np.sum(is_ripple), np.sum(is_ripple) / float(len(is_ripple))

        print df
        fpath = os.path.join('csv', 'uses.pkl')
        df.to_pickle(fpath)
        fpath = os.path.join('csv', 'uses.csv')
        df.to_csv(fpath)


    def format_rankings(self):

        rankings = {20: '3>2>1', 22:'', 24:'3>2>1', 34: '1>3>2',
             43:'', 355:'2 > 3 > 1', 365:'3>2>1', 444:'3>1>2',
             446:'2 > 3 > 1'}

        df = pd.DataFrame(columns = self.condition_ids)
        count = 0
        for user_id in self.complete_entries_by_conditions_keys:
            user_rankings = rankings[user_id].strip()
            if len(user_rankings) > 1:
                parts = user_rankings.split('>')
                rs = [int(part.strip())-1 for part in parts]

                ordering = self.ordering_by_user[user_id]
                ordering = [int(idx) for idx in ordering]
                ranks = {}
                for i, experiment_idx in enumerate(rs):
                    condition_idx = ordering[experiment_idx]
                    condition_id = self.condition_ids[condition_idx]
                    ranks[condition_id] = i

                df.loc[count] = [ranks[condition_id] for condition_id in self.condition_ids]
                count += 1

        fpath = os.path.join('csv', 'ranks.csv')
        df.to_pickle(fpath)



    def collect_best_seq_likelihood(self):
        self.likelihoods_by_condition = {}
        for user_id in self.complete_entries_by_conditions_keys:
            ll = {}
            for condition_id, seq_entry in self.best_seq_by_condition[user_id].iteritems():
                ll[condition_id] = self.ngram.log_likelihood(seq_entry[1])
            self.likelihoods_by_condition[user_id] = ll

    def collect_best_seq_mean_novelty(self):
        self.unigram_inverse_freq = {}

        for user_id in self.complete_entries_by_conditions_keys:
            novelties_condition = {}
            for condition_id, seq_entry in self.best_seq_by_condition[user_id].iteritems():
                seq = seq_entry[1]
                novelties = []
                for sym in seq:
                    inverse_freq = self.ngram.unigram_inverse_freq(sym)
                    if inverse_freq is not None:
                        novelties.append(inverse_freq)
                mean = np.mean(novelties)

                novelties_condition[condition_id] = {'mean': mean,
                                                     'log_mean': np.log(mean)}

            self.unigram_inverse_freq[user_id] = novelties_condition

    def rescale(self, items):
        min = np.min(items)
        max = np.max(items)
        range_ = max - min
        return min, range_

    def compute_goodness_cross_novelty(self):
        novelties = []
        goodnesses = []
        for user_id in self.complete_entries_by_conditions_keys:
            for condition_id in self.condition_ids:
                novelty = self.unigram_inverse_freq[user_id][condition_id]['log_mean']
                goodness = self.best_seq_ratings_by_condition[user_id][condition_id]
                novelties.append(novelty)
                goodnesses.append(goodness)

        novelty_min, novelty_range = self.rescale(novelties)
        goodness_min, goodness_range = self.rescale(goodnesses)

        self.novelty_goodness_by_condition = {}
        for user_id in self.complete_entries_by_conditions_keys:
            user_novelty_goodness = {}
            for condition_id in self.condition_ids:
                novelty = self.unigram_inverse_freq[user_id][condition_id]['log_mean']
                goodness = self.best_seq_ratings_by_condition[user_id][condition_id]
                novelty_scaled = (novelty - novelty_min) / novelty_range
                goodness_scaled = (goodness - goodness_min) / goodness_range
                novelty_goodness_prod = novelty_scaled * goodness_scaled
                novelty_goodness_average = (novelty_scaled + goodness_scaled) / 2
                res = dict(novelty_scaled=novelty_scaled,
                           goodness_scaled=goodness_scaled,
                           novelty_goodness_prod=novelty_goodness_prod,
                           novelty_goodness_average=novelty_goodness_average)
                user_novelty_goodness[condition_id] = res
            self.novelty_goodness_by_condition[user_id] = user_novelty_goodness


    def print_best_seq_rating_and_likelihood(self):
        for user_id in self.complete_entries_by_conditions_keys:
            for condition_id in self.condition_ids:
                seq = self.best_seq_by_condition[user_id][condition_id][1]
                seq_str = ' '.join(seq)
                ll = self.likelihoods_by_condition[user_id][condition_id]
                print '%s, %s, %.2f, %.1f, %s' % (user_id, condition_id, \
                                                ll, self.best_seq_ratings_by_condition[user_id][condition_id], \
                                                seq_str)

    def print_results(self):
        novelties = self.results['best_seq-log_mean_unigram_inverse_freq']
        ratings = self.results['best_seq-rating']
        lls = self.results['best_seq-loglikelihood']
        conditions = self.results['condition']
        seqs = self.results['best_seq']
        user_id = self.results['user_id']

        sorted_inds = np.argsort(novelties)
        for idx in sorted_inds:
            seq = seqs[idx]
            print conditions[idx],
            print user_id[idx],
            print ', %.3f, %.2f, %.3f,' % (novelties[idx], ratings[idx], lls[idx]),
            print ' '.join(seq), ', '
            # print '\t',
            # for sym in seq:
            #     score = self.ngram.unigram_inverse_freq(sym)
            #     if score is not None:
            #         print '%.3f,' % score,
            #     else:
            #         print 'None',
            print

    def collect_experiment_ordering(self, add=True):
        ordering_by_order = defaultdict(list)
        ordering_by_user = defaultdict(list)
        for user_id in self.complete_entries_by_conditions_keys:
            entries = self.complete_entries_by_conditions[user_id]
            ordering_entries = self.filter_actionKind('ordering', entries)
            # print 'ordering_entries', ordering_entries
            assert len(ordering_entries) == 1
            user_ordering = list(ordering_entries[0]["data"])
            ordering_by_user[user_id] = user_ordering
            for i in range(3):
                ordering_by_order[i].append(user_ordering[i])
        if add:
            for i in range(3):
                self.results['ordering-%d' % i] = ordering_by_order[i]
        self.ordering_by_user = ordering_by_user
        print self.results

    def filter_actionKind(self, actionKind, entries):
        filtered_entries = []
        for condition_entries in entries.values():
            for entry in condition_entries:
                if entry['actionKind'] == actionKind:
                    filtered_entries.append(entry)
        return filtered_entries

    def is_chord_seq(self, entry):
        seq = entry['seq']
        for sym in seq:
            if len(sym) >= 9:
                return False
        return True

    def collect_best_rating_seq_by_condition(self, add=True):
        best_seq_by_condition = {}
        final_seqs_by_condition = {}
        for user_id in self.complete_entries_by_conditions_keys:
            print '---', user_id
            conditions = self.complete_entries_by_conditions[user_id]

            user_best_seq_by_condition = {}
            for condition_id in self.condition_ids:
                saved_seq_entries = []
                entries = conditions[condition_id]
                rating_score = []
                print condition_id
                for entry in entries:
                    if entry["actionKind"] == "rating" and \
                                    len(entry['seq']) == CHORD_LEN:
                        if self.is_chord_seq(entry):
                            print condition_id, entry['rating'], entry['seq']
                            saved_seq_entries.append(entry)
                            rating_score.append(float(entry['rating']))
                        # no false negatives
                        # else:
                        #     print condition_id, entry['seq']
                final_seqs_by_condition[condition_id] = saved_seq_entries
                max_idx = np.argmax(rating_score)

                user_best_seq_by_condition[condition_id] = [rating_score[max_idx],
                                                           saved_seq_entries[max_idx]['seq']]
            best_seq_by_condition[user_id] = user_best_seq_by_condition

        best_seq_ratings_by_condition = {}
        for condition_id in self.condition_ids:
            ratings = []
            for user_id in self.complete_entries_by_conditions_keys:
                if user_id not in best_seq_ratings_by_condition:
                    best_seq_ratings_by_condition[user_id] = {}
                best_rating = best_seq_by_condition[user_id][condition_id][0]
                ratings.append(best_rating)
                best_seq_ratings_by_condition[user_id][condition_id] = best_rating
            column_label = '%s-best_seq_ratings_by_condition' % condition_id
            if add:
                self.results[column_label] = ratings

        print self.results
        self.best_seq_ratings_by_condition = best_seq_ratings_by_condition
        self.best_seq_by_condition = best_seq_by_condition

        max_counts = defaultdict(lambda: 0)
        for user_id in self.complete_entries_by_conditions_keys:
            best_seqs = best_seq_by_condition[user_id]
            best_scores = [best_seqs[condition_id][0] for condition_id in self.condition_ids]
            print 'best_scores', best_scores
            max_score = np.max(best_scores)
            for i, score in enumerate(best_scores):
                if score == max_score:
                    ex_type = self.condition_ids[i]
                    print max_score, ex_type,
                    max_counts[ex_type] += 1

        print
        for condition in self.condition_ids:
            print condition, max_counts[condition]

            # max_idx = np.argmax(best_scores)
            # print self.condition_ids[max_idx], best_scores[max_idx]


    def filter_complete_user_entries(self):
        complete_count = 0
        complete_entries = {}
        for user_id, entries in self.user_entries.iteritems():
            # filter out comments...
            entries_filtered = self.filter_entries_to_only_chord_seqs(entries)
            if user_id in USER_ID_IGNORE_LIST:
                continue
            for i, entry in enumerate(entries_filtered):
                if entry['actionKind'] == 'comments':
                    continue
                seq = entry['seq_formatted']
                entry_complete = self.is_complete_seq(seq)

                if entry_complete and not self.is_test(entries_filtered)\
                        and self.is_valid_seq(seq):
                    complete_count += 1
                    complete_entries[user_id] = entries_filtered

                    # print '---key', user_id
                    # for entry in entries_filtered:
                    #     print entry["seq_formatted"]

                    break

        self.complete_entries = complete_entries
        print '# of entries:', len(self.user_entries)
        print '# of complete_entires:', len(complete_entries)
        print complete_entries.keys()
        self.complete_user_keys = Keys(complete_entries.keys(), 'all_users')
        # for entries in complete_entries:
        #     print self.last_complete_seq(entries)
        # return complete_count

    def filter_last_complete_seqs(self):
        last_complete_seqs = {}
        for key in self.complete_user_keys:
            # print '---key', key
            seq = self.last_complete_seq(self.complete_entries[key])
            assert seq is not None
            last_complete_seqs[key] = seq
        self.last_complete_seqs = last_complete_seqs

    def filter_first_complete_seqs(self):
        seqs = {}
        reached_time_in_percentage = {}
        entries = self.chord_change_entries
        for key in self.complete_user_keys:
            seq, reached_idx = \
                self.first_complete_seq(entries[key])
            seqs[key] = seq
            reached_time = reached_idx / float(len(entries))
            reached_time_in_percentage[key] = reached_time
            # TODO: remove hack, this idx corresponds to chord_change_entry idx, not complete_entry idx
            self.complete_entries[key][0]["reached_first_complete_idx"] = reached_idx

        self.first_complete_seqs = seqs
        self.reached_time_in_percentage = reached_time_in_percentage

        #reached_time_in_percentage_collapsed = self.collapse_all(reached_time_in_percentage)
        self.plot_dist(reached_time_in_percentage, 'reached_time_in_percentage')

    def all_remaining_unknown(self, seq, i):
        for sym in seq[i:]:
            if sym != UNKNOWN_EMPTY_SYMBOL:
                return False
        return True


    def print_seq(self, seq):
        for i, sym in enumerate(seq[:CHORD_LEN]):
            if not self.all_remaining_unknown(seq, i) and sym == UNKNOWN_EMPTY_SYMBOL:
                print '--',
            elif sym == UNKNOWN_EMPTY_SYMBOL:
                pass
            else:
                print sym,
        print


    def _last_seq_novelty(self, keys=None):
        if keys is None:
            keys = self.complete_user_keys
        novelties = {}
        novelties_list = []
        for key in keys:
            seq = self.last_complete_seqs[key]
            # actually inverse, the smaller the likelihood, the more novel
            novelty = self.ngram.log_likelihood(seq)
            novelties_list.append(novelty)
            novelties[key] = novelty

        # self.plot_dist(novelties, 'last_seq_novelty-%s' % keys.name)

        sorted_indices = np.argsort(novelties_list)
        # the less is more creative
        print '--- first ones are the most creative ---'
        for idx in sorted_indices:
            key = keys[idx]
            print '%.2f' % novelties[key],
            self.print_seq(self.last_complete_seqs[key])
        print '--- end least creative'
        self.last_seq_novelties = novelties

        # not doing the inverse earlier because want to preserve likelihood for other plots
        self.last_seq_novelties_inversed = {}
        for key, novelty in novelties.iteritems():
            self.last_seq_novelties_inversed[key] = 1.0/novelties[key]

        return novelties

    def filter_num_unique_chords(self, keys):
        num_chords = self.num_unique_chords.values()
        threshold = np.mean(num_chords) + 2*np.std(num_chords)
        print np.mean(num_chords), threshold
        filtered_keys = [key for key in keys if self.num_unique_chords[key] < threshold]
        print 'key size, before, after', len(keys), len(filtered_keys)
        return Keys(filtered_keys, 'filtered_' + keys.name)

    def process_against_final(self, keys=None):
        if keys is None:
            keys = self.complete_user_keys
        filtered_keys = self.filter_num_unique_chords(keys)
        # if behavior in exploration influences end-product novelty?

        novelties_subset = [self.last_seq_novelties_inversed[key] for key in filtered_keys]

        # num of unique chords explored
        num_unique_chords_subset = [self.num_unique_chords[key] for key in filtered_keys]

        plot_id = 'unique-final_novelty-%s' % filtered_keys.name
        self.plot_linear_regression(num_unique_chords_subset, novelties_subset,
                                    'num_unique_chords', 'last_seq_novelty', plot_id)

        # num of chord changes
        num_chord_changes_subset = [len(self.chord_changes_collapsed[key]) for key in filtered_keys]
        print 'num_chord_changes_subset', num_chord_changes_subset
        plot_id = 'chord_change_count-final_novelty-%s' % filtered_keys.name
        self.plot_linear_regression(num_chord_changes_subset, novelties_subset,
                                    'num_chord_changes', 'last_seq_novelty', plot_id)

        self.process_regressions_to_final_novelty('top_novelty_in_changes', filtered_keys, novelties_subset)
        self.process_regressions_to_final_novelty('novelty_std_in_changes', filtered_keys, novelties_subset)

    def process_regressions_to_final_novelty(self, func_name, filtered_keys, novelties_subset):
        subset = [getattr(self, func_name)[key] for key in filtered_keys]
        print_vector(subset, func_name, truncate=False)
        plot_id = '%s-final_novelty-%s' % (func_name, filtered_keys.name)
        self.plot_linear_regression(subset, novelties_subset,
                                    func_name, 'last_seq_novelty', plot_id)

    def _novelty_std_in_changes(self):
        novelty_in_changes = self.collapse_steps(self.likelihood_by_step,
                                                 skip_steps=[0])
        self.novelty_std_in_changes = {}
        for key, novelties in novelty_in_changes.iteritems():
            self.novelty_std_in_changes[key] = np.std(novelties)

    def _top_novelty_in_changes(self):
        novelty_in_changes = self.collapse_steps(self.likelihood_by_step,
                                                 skip_steps=[0])
        self.top_novelty_in_changes = {}
        for key, novelties in novelty_in_changes.iteritems():
            self.top_novelty_in_changes[key] = np.max(novelties)

    def count_diffs(self, seq, seq_other):
        diff_count = 0
        length = len(seq)
        # want shorter one
        if len(seq) > len(seq_other):
            length = len(seq_other)
        for i in range(length):
            if seq[i] != seq_other[i]:
                diff_count += 1
        return diff_count

    def is_ripple(self, entry):
        chordSeqsAndFormat = entry["chordSeqsAndFormat"]
        count_change = 0
        print 'chordSeqsAndFormat', chordSeqsAndFormat
        for format in chordSeqsAndFormat:
            if format[1]:
                count_change += 1
        if count_change == 3:
            return True
        else:
            return False
        
    def get_changed_indices(self, entry):
        chordSeqsAndFormat = entry["chordSeqsAndFormat"]
        changed_inds = []
        for i, format in enumerate(chordSeqsAndFormat):
            if format[1]:
                changed_inds.append(i)
        # check if contiguous
        if len(changed_inds) == 1:
            return changed_inds

        # just for ripples, not side ripples, which would be non-contiguous
        # and of length 2
        if len(changed_inds) == 3:
            for i in range(len(changed_inds)-1):
                print i, changed_inds
                assert changed_inds[i] + 1 == changed_inds[i+1]

        return changed_inds

    def is_contextual_ripple(self, entry):
        chordSeqsAndFormat = entry["chordSeqsAndFormat"]
        num_seq = len(chordSeqsAndFormat)
        changed_inds = self.get_changed_indices(entry)
        # examples that are not contextual ripples
        # [[u'Cm', False], [u'F', True]]
        #
        if changed_inds[0] != 0 or changed_inds[-1] < num_seq:
            return True
        else:
            return False

    def collect_use_dependencies(self):
        # needed to have run collect_use_actions
        print self.complete_entries.values()[0][0].keys()
        assert 'ripple' in self.complete_entries.values()[0][0].keys()

        # collect as pairs
        use_actions = defaultdict(list)
        for key in self.complete_user_keys:
            entries = self.complete_entries[key]
            for i, entry in enumerate(entries):
                if i + 1 < len(entries) and \
                        entries[i+1]['ripple']:
                    seq = entry['seq_formatted']
                    seq_next = entries[i+1]['actionKind']


                    use_actions[key].append([entry, entries[i+1]])
                    print '--- pair'
                    print i, entry['actionKind'], entry['seq_formatted'], entry['seq']
                    print i+1, entries[i+1]['actionKind'], entries[i+1]['seq_formatted'], entries[i+1]['seq']
                    # print i+2, entries[i+2]['actionKind'], entries[i+2]['seq_formatted']

        self.use_action_dependency_pair_entries = use_actions

    def analysze_use_actions(self):
        num_used_actions = {}
        for key in self.complete_user_keys:
            num_used_actions[key] = len(self.use_action_entries[key])
        print 'num_used_actions', num_used_actions.values()
        self.plot_dist(num_used_actions, "num_used_actions")

        percentage_ripple = {}
        num_ripples = {}
        percentage_ripple_after = {}
        for key in self.complete_user_keys:
            count = 0
            after_first_complete_ripple_count = 0
            entries = self.use_action_entries[key]
            for i, entry in enumerate(entries):
                if entry["ripple"]:
                    count += 1
                    if i > self.complete_entries[key][0]["reached_first_complete_idx"]:
                        after_first_complete_ripple_count += 1
            num_ripples[key] = count
            if len(entries) > 0:
                percentage_ripple[key] = float(count) / len(entries)
            else:
                percentage_ripple[key] = 0

            if count > 0:
                percentage_ripple_after[key] = float(after_first_complete_ripple_count) / count
            else:
                percentage_ripple_after[key] = -1
        print 'percentage_ripple', percentage_ripple.values()
        print 'num_ripples', num_ripples.values()

        self.plot_dist(percentage_ripple, "percentage_ripple")

        percentage_ripple_after = np.asarray(percentage_ripple_after.values())
        filtered_per_ripple_after = percentage_ripple_after[percentage_ripple_after>=0]
        print "percentage_ripple_after", percentage_ripple_after
        print "filtered_per_ripple_after", list(filtered_per_ripple_after)

        self.plot_dist(filtered_per_ripple_after, "percentage_ripple_after")

        # for people who did use ripples, is ripple used more often for more novel chords?
        ripple_novelty_for_users_that_use_ripples = {}
        singleton_novelty_for_users_that_use_ripples = {}
        ripple_user_keys = []
        for key in self.complete_user_keys:
            if percentage_ripple[key] > 0:
                ripple_user_keys.append(key)
                ripple_novelties = []
                singleton_novelties = []
                for entries in self.use_action_entries.values():
                    for entry in entries:
                        if entry["ripple"]:
                            novelty = self.novelty_of_ripple_middle_chord(entry)
                            if novelty is not None:
                                ripple_novelties.append(novelty)
                        else:
                            novelty = self.novelty_of_singleton(entry)
                            if novelty is not None:
                                singleton_novelties.append(novelty)
                ripple_novelty_for_users_that_use_ripples[key] = ripple_novelties
                singleton_novelty_for_users_that_use_ripples[key] = singleton_novelties

        print '# of ripple users', len(ripple_user_keys)
        print 'average ripple_novelties', np.mean(ripple_novelties), len(ripple_novelties), ripple_novelties
        print 'average singleton_novelties', np.mean(singleton_novelties), len(singleton_novelties), singleton_novelties
        # of ripple users 11, out of 22
        # average ripple_novelties 0.0567481246487 50
        # singleton_novelties 0.178724476307 61

        ripple_user_ripple_percentage = {}
        for key in ripple_user_keys:
            ripple_user_ripple_percentage[key] = percentage_ripple[key]

        xs = self.get_values(ripple_user_ripple_percentage, ripple_user_keys)
        ys = [self.last_seq_novelties_inversed[key] for key in ripple_user_keys]

        plot_id = 'ripple_percent-final_novelty'
        self.plot_linear_regression(xs, ys, 'ripple_percent', 'last_seq_novelty',
                                    plot_id)

    def collect_use_actions(self):
        use_actions = defaultdict(list)
        diff_counts = []
        for key in self.complete_user_keys:
            entries = self.complete_entries[key]
            # entries = self.chord_change_entries[key]
            for i, entry in enumerate(entries):
                #print entry.keys()
                if 'use' in entry["actionKind"] and \
                        entry["author"] == 'machine':
                    # TODO: hack, just storing in first position
                    if i <= self.complete_entries[key][0]["reached_first_complete_idx"]:
                        entry["before_complete"] = True
                    else:
                        entry["before_complete"] = False

                    if self.is_ripple(entry):
                        entry['ripple'] = True
                    else:
                        entry['ripple'] = False
                    use_actions[key].append(entry)
                else:
                    entry['ripple'] = False
        self.use_action_entries = use_actions

        print "diff_counts", diff_counts

    def collect_likelihood_versus_is_ripple(self):
        # collapsed all users
        # TODO: could add a column that includes users
        # TODO: and also add a column that's the likelihood
        likelihood_vs_is_ripple = []
        for key, entries in self.use_action_entries.iteritems():
            for entry in entries:
                print '------ entry'
                chordSeqsAndFormat = entry["chordSeqsAndFormat"]
                print "chordSeqsAndFormat", chordSeqsAndFormat
                seq = entry["seq_formatted"]
                print 'seq', seq
                is_contextual_ripple = self.is_contextual_ripple(entry)
                print 'is_contextual_ripple', is_contextual_ripple

                novelty, novelty_original = self.novelty_of_ripple_middle_chord(entry)
                if novelty is None or novelty_original is None:
                    continue
                if novelty == novelty_original:
                    continue
                if novelty == 1.0:
                    continue
                if entry['ripple']:
                    is_ripple = 1
                else:
                    is_ripple = 0
                data = [novelty, novelty_original, is_ripple, key]
                print data
                likelihood_vs_is_ripple.append(data)

        fname = 'likelihood_vs_is_ripple.csv'
        with open(fname, "wb") as f:
            writer = csv.writer(f)
            writer.writerows(likelihood_vs_is_ripple)


    def get_values(self, dictionary, key_ordering):
        return [dictionary[key] for key in key_ordering]


    def novelty_of_ripple_middle_chord(self, entry):
        chordSeqsAndFormat= entry["chordSeqsAndFormat"]
        # print "chordSeqsAndFormat", chordSeqsAndFormat
        seq = entry["seq_formatted"]
        # print 'seq', seq
        assert len(seq) == len(chordSeqsAndFormat)
        middle_idx = None
        first_change = True
        for i, format in enumerate(chordSeqsAndFormat):
            if format[1] and first_change:
                first_change = False
                # set for singleton
                middle_idx = i
            elif format[1] and not first_change:
                middle_idx = i
                break
        # print "chordSeqsAndFormat", chordSeqsAndFormat
        # print "middle_idx", middle_idx
        subseq = self.merge_seq(chordSeqsAndFormat, middle_idx, seq)
        if subseq is None:
            assert len(seq) == 1, middle_idx
            return None
        novelty = self.ngram.log_likelihood(subseq)
        original_subseq = self.original_seq_at_change(middle_idx, seq)
        novelty_original = self.ngram.log_likelihood(original_subseq)
        return novelty, novelty_original

    def merge_seq_complete(self, chordSeqsAndFormat, middle_idx, seq):
        # print len(chordSeqsAndFormat), len(seq), middle_idx
        if len(seq) > middle_idx+1 and middle_idx-1 >= 0:
            subseq = [seq[middle_idx-1], chordSeqsAndFormat[middle_idx][0], seq[middle_idx+1]]
        elif middle_idx-1 >= 0:
            subseq = [seq[middle_idx-1], chordSeqsAndFormat[middle_idx][0]]
        elif len(seq) > middle_idx+1:
            subseq = [chordSeqsAndFormat[middle_idx][0], seq[middle_idx+1]]
        else:
            subseq = [chordSeqsAndFormat[middle_idx][0]]
        return

    def original_seq_at_change(self, middle_idx, seq):
        # print len(chordSeqsAndFormat), len(seq), middle_idx
        if len(seq) > middle_idx+1 and middle_idx-1 >= 0:
            subseq = [seq[middle_idx-1], seq[middle_idx], seq[middle_idx+1]]
        elif middle_idx-1 >= 0:
            subseq = [seq[middle_idx-1], seq[middle_idx]]
        elif len(seq) > middle_idx+1:
            subseq = [seq[0], seq[middle_idx+1]]
        else:
            subseq = [seq[middle_idx]]
        return subseq

    def merge_seq(self, chordSeqsAndFormat, middle_idx, seq):
        # print len(chordSeqsAndFormat), len(seq), middle_idx
        if len(seq) > middle_idx+1 and middle_idx-1 >= 0:
            subseq = [seq[middle_idx-1], chordSeqsAndFormat[middle_idx][0], seq[middle_idx+1]]
        elif middle_idx-1 >= 0:
            subseq = [seq[middle_idx-1], chordSeqsAndFormat[middle_idx][0]]
        elif len(seq) > middle_idx+1:
            subseq = [chordSeqsAndFormat[middle_idx][0], seq[middle_idx+1]]
        else:
            subseq = [chordSeqsAndFormat[middle_idx][0]]
        return subseq
    
    def novelty_of_singleton(self, entry):
        chordSeqsAndFormat= entry["chordSeqsAndFormat"]
        seq = entry["seq_formatted"]
        assert len(seq) == len(chordSeqsAndFormat)
        change_idx = None
        for i, format in enumerate(chordSeqsAndFormat):
            if format[1]:
                change_idx = i
                break
        subseq = self.merge_seq(chordSeqsAndFormat, change_idx, seq)
        if subseq is None:
            return None
        novelty = self.ngram.log_likelihood(subseq)
        return novelty
    
    def is_singleton(self, entry):
        chordSeqsAndFormat= entry["chordSeqsAndFormat"]
        seq = entry["seq_formatted"]
        assert len(seq) == len(chordSeqsAndFormat)
        changed_counts = 0
        for i, format in enumerate(chordSeqsAndFormat):
            if format[1]:
                changed_counts += 1
        if changed_counts == 1:
            return True
        else:
            return False
        
    def collapse_all(self, by_steps):
        if not len(by_steps):
            return []
        if len(by_steps) > 0 and isinstance(by_steps.values()[0], dict):
            by_steps = self.collapse_steps(by_steps)
        vals = []
        for user_num, val in by_steps.iteritems():
            if isinstance(val, list):
                vals.extend(val)
            else:
                vals.append(val)
        return vals

    def collapse_steps(self, by_steps, skip_steps=[]):
        if not isinstance(by_steps.values()[0], dict):
            return by_steps
        collapsed = {}
        for key, step_dict in by_steps.iteritems():
            flattened = []
            for step, vals in step_dict.iteritems():
                if step not in skip_steps and step < CHORD_LEN:
                    flattened.extend(vals)
            collapsed[key] = flattened
        return collapsed

    def collapse_users(self, by_steps):
        if not isinstance(by_steps.values()[0], dict):
            return by_steps
        collapsed = defaultdict(list)
        print 'by_steps', by_steps
        for key, step_dict in by_steps.iteritems():
            for step, vals in step_dict.iteritems():
                collapsed[step].extend(vals)
        return collapsed

    def plot_linear_regression(self, xs, ys, xlabel, ylabel, plot_id):
        sns.set(color_codes=True)
        data = np.asarray([xs, ys]).T

        df = pd.DataFrame(data=data,
                          columns=[xlabel, ylabel])

        sns.plt.clf()
        sns.lmplot(x=xlabel, y=ylabel, data=df)
        plt.savefig(os.path.join(PLOT_DIR, 'regression-%s.pdf' % plot_id))


    def time_spent(self):
        time = []
        threshold = 600  # 6 minutes
        self.casual_user_keys = Keys([], 'causal_user')
        self.serious_user_keys = Keys([], 'serious_user')

        for user_num in self.complete_user_keys:
            entries = self.complete_entries[user_num]
            start_time = entries[0]['time']
            end_time = entries[-1]['time']
            diff = end_time - start_time
            time.append(diff)
            if diff >= threshold:
                self.serious_user_keys.append(user_num)
            else:
                self.casual_user_keys.append(user_num)

        print 'Time spent (in minutes): %.2f, %.2f' % \
              (np.mean(time)/60, np.std(time)/60)

        # sns.plt.clf()
        # sns.distplot(time, kde=False, color="b")
        # fpath = os.path.join(PLOT_DIR, 'time.pdf')
        # plt.savefig(fpath)


    # def collect_chords_used_collaspsed(self):
    #     self.chords_used_collapsed = {}
    #     for user_num, entries in self.complete_entries.iteritems():
    #         syms = []
    #         for entry in entries:
    #             seq = entry['seq_formatted']
    #             # print seq
    #             syms.extend(seq)
    #         non_empty_syms = [sym for sym in syms if sym != 'Unknown']
    #         # valid_syms = self.validate_chord_symbol(non_empty_syms)
    #         self.chords_used_collapsed[user_num] = non_empty_syms

    def collect_chord_changes_by_step(self):
        # step being the chord idx
        # collect all changes first, not unique
        self.chord_changes_by_step = {}
        for user_num, entries in self.complete_entries.iteritems():
            # per user
            chord_changes_by_step = defaultdict(list)
            for entry in entries:
                # per action
                seq = entry['seq_formatted']
                # print seq
                # print 'len(seq)', len(seq)
                for i, sym in enumerate(seq):
                    if len(seq) > CHORD_LEN:
                        continue
                    # only if sym is not different from previous sym
                    if len(chord_changes_by_step[i]) == 0 \
                            or chord_changes_by_step[i][-1] != sym:
                        chord_changes_by_step[i].append(sym)
            self.chord_changes_by_step[user_num] = chord_changes_by_step

        self.chord_changes_collapsed = {}
        for key, chord_changes_by_step in self.chord_changes_by_step.iteritems():
            chord_changes_collapased = []
            for chord_changes in chord_changes_by_step.values():
                chord_changes_collapased.extend(chord_changes)
            self.chord_changes_collapsed[key] = chord_changes_collapased

    def is_valid(self, sym):
        if sym in self.valid_syms:
            return True
        elif sym in self.not_valid_syms:
            return False
        else:
            try:
                formatted_sym = preprocess_letters_before_sym2chord(sym)
                harmony.ChordSymbol(formatted_sym)
                return True
            except:
                print 'WARNING: %s not a valid chord symbol for music21' % sym
                return False

    def is_valid_seq(self, seq):
        # if more than 3 not recognized then not valid?
        invalid_counts = 0
        syms_known_by_model = self.nn.syms
        for sym in seq:
            formatted_sym, success = self.validate_chord_symbol(sym)
            if not success or sym not in syms_known_by_model:
                invalid_counts += 1
                if invalid_counts > 3:
                    return False
        return True

    def validate_chord_symbols(self, syms):
        valid_syms_local = []
        for sym in syms:
            sym, valid = self.validate_chord_symbol(sym)
            if valid:
                valid_syms_local.append(sym)
        return valid_syms_local

    def validate_chord_symbol(self, sym, format=True):
        if sym in self.valid_syms:
            if format:
                return self.valid_syms[sym], True
            else:
                return sym, True
        elif sym in self.not_valid_syms:
            return sym, False
        else:
            try:
                formatted_sym = preprocess_letters_before_sym2chord(sym)
                harmony.ChordSymbol(formatted_sym)
                if sym not in self.valid_syms:
                    self.valid_syms[sym] = formatted_sym

                if format:
                    return formatted_sym, True
                else:
                    return sym, False
            except:
                print 'WARNING: %s not a valid chord symbol for music21' % sym
                self.not_valid_syms.append(sym)
                return sym, False

    def _unique_chords(self, keys=None):
        if keys is None:
            keys = self.complete_user_keys

        num_chords = {}
        for key in keys:
            all_syms = self.chord_changes_collapsed[key]
            syms = list(set(all_syms))
            # num_chords.append(len(syms))
            num_chords[key] = len(syms)
        print '# of unique chords:  %.2f, %.2f' % \
              (np.mean(num_chords.values()), np.std(num_chords.values()))

        # sns.plt.clf()
        # sns.distplot(num_chords, kde=True, color="b")
        # fpath = os.path.join(PLOT_DIR, 'unique_chords.pdf')
        # plt.savefig(fpath)
        self.num_unique_chords = num_chords
        return num_chords

    def plot_dist(self, one_d_data, plot_id):
        if isinstance(one_d_data, dict):
            print '%s: %.2f, %.2f' % \
                  (plot_id, np.mean(one_d_data.values()), np.std(one_d_data.values()))
        else:
            print 'one_d_data', one_d_data
            print '%s: %.2f, %.2f' % \
                  (plot_id, np.mean(one_d_data), np.std(one_d_data))

        # np.mean(one_d_data)/60, np.std(one_d_data)/60)
        sns.set()
        sns.plt.clf()
        num_bins = 10
        if isinstance(one_d_data, dict):
            sns.distplot(one_d_data.values(), bins=num_bins, kde=False, color="b")
        else:
            sns.distplot(one_d_data, bins=num_bins, kde=False, color="b")
        fpath = os.path.join(PLOT_DIR, '%s.pdf' % plot_id)
        plt.savefig(fpath)

    def summarize_chord_changes_by_step(self, keys):
        num_chord_changes_by_step = defaultdict(list)
        num_unique_chord_changes_by_step = defaultdict(list)
        num_hesitations = defaultdict(list)
        for key in keys:
            changes = self.chord_changes_by_step[key]
            # print 'changes', changes
            for i, chords in changes.iteritems():
                # print 'chords', chords
                unique_chords = list(set(chords))
                hesitation_counts = len(chords) - len(unique_chords)

                num_chord_changes_by_step[i].append(len(chords))
                num_unique_chord_changes_by_step[i].append(len(unique_chords))
                num_hesitations[i].append(hesitation_counts)

        self.print_stats_by_step(num_chord_changes_by_step)
        self.print_stats_by_step(num_unique_chord_changes_by_step)

        self.plot_stats_by_step(num_chord_changes_by_step,
                                num_unique_chord_changes_by_step,
                                'num_chord_changes_by_step')

        self.plot_stats_by_step_single(num_hesitations, 'hesitations')

    def get_novelty_threshold(self, novelty_by_step):
        # flatten
        novelties = self.collapse_all(novelty_by_step)
        # print 'novelties', novelties
        novelties = np.asarray(novelties)
        # print novelties[novelties>100]
        # print np.max(novelties[novelties<=100])

    def summarize_novelty_by_step(self, keys=None):
        # by step by user
        if keys is None:
            keys = self.complete_user_keys

        likelihood_by_step = defaultdict(list)
        # TODO: still need to make theta_by_step to have users
        theta_by_step = defaultdict(list)
        for key in keys:
            theta_by_step_user = defaultdict(list)
            likelihood_by_step_user = defaultdict(list)
            for entry in self.complete_entries[key]:
                seq = entry['seq_formatted']
                # print 'seq_formatted', seq
                # print 'seq[0]', seq[0]
                if len(seq) > 0 and seq[0] != UNKNOWN_EMPTY_SYMBOL and \
                        seq[0] in self.ngram.syms:
                    novelty = self.ngram.log_likelihood([seq[0]])
                    likelihood_by_step_user[0].append(novelty)
                    theta = self.nn.theta(seq[0])
                    theta_by_step_user[0].append(theta)
                for i in range(1, len(seq)):
                    if i >= CHORD_LEN:
                        continue
                    # print seq[i-1:i+1]
                    both_in_sym = seq[i-1] in self.ngram.syms and \
                                        seq[i] in self.ngram.syms
                    both_valid = seq[i-1] != UNKNOWN_EMPTY_SYMBOL and \
                                        seq[i] != UNKNOWN_EMPTY_SYMBOL
                    if both_in_sym and both_valid:
                        novelty = self.ngram.log_likelihood(seq[i-1:i+1])
                        theta_by_step_user[i].append(novelty)
                        likelihood_by_step_user[i].append(novelty)

                    if seq[i] != UNKNOWN_EMPTY_SYMBOL and seq[i] in self.nn.syms:
                        theta = self.nn.theta(seq[i])
                        theta_by_step_user[i].append(theta)

            theta_by_step[key] = theta_by_step_user
            likelihood_by_step[key] = likelihood_by_step_user

        # print '---novelty---'
        # for key, val in novelty_theta_by_step.iteritems():
        #     print key, val

        self.theta_by_step = theta_by_step
        self.likelihood_by_step = likelihood_by_step

        # print '--- get_novelty_threshold  ---'
        # TODO: not sure this is getting at novelty
        self.get_novelty_threshold(theta_by_step)

        self.print_stats_by_step(theta_by_step)
        self.plot_stats_by_step_single(theta_by_step, 'novelty')

        # print '---theta---'
        # for key, val in novelty_theta_by_step.iteritems():
        #     print key, val
        # self.print_stats_by_step(theta_by_step)
        # # TODO: theta not yet a nested dictionary
        # self.plot_stats_by_step_single(theta_by_step, 'theta')
        return theta_by_step

    def theta_as_time_series(self, keys):
        print '---theta_as_time_series---', keys
        theta_seqs = []
        for key in keys:
            print '--- user', key
            seq = self.last_complete_seq(self.complete_entries[key])
            print 'last complete seq', seq
            if seq is None:
                continue
            seq = seq[:CHORD_LEN]
            thetas = []
            for sym in seq:
                theta = self.nn.theta(sym)
                if theta is None:
                    # theta = np.nan
                    theta = 0
                thetas.append(theta)
            print thetas
            theta_seqs.append(thetas)
        print '# of theta seqs:', len(theta_seqs)

        sns.set(style="darkgrid", palette="Set2")

        # Plot the average over replicates with bootstrap resamples
        sns.tsplot(np.asarray(theta_seqs), err_style="boot_traces", n_boot=500)
        plt.savefig(os.path.join(PLOT_DIR, 'timeseries.pdf'))


    def print_stats_by_step(self, num_chord_changes_by_step):
        # print '---------------'
        # for changes in num_chord_changes_by_step:
        #     print changes
        mean = [ np.mean(changes) for changes in num_chord_changes_by_step]
        std = [ np.std(changes) for changes in num_chord_changes_by_step]
        print '# of chord changes in position:', len(mean)
        print_vector(mean, 'mean', False)
        print_vector(std, 'std', False)

    def plot_stats_by_step_single(self, num_changes, plot_id):
        num_changes = self.collapse_users(num_changes)
        flattened = []
        for step_idx, counts in num_changes.iteritems():
            for j, count in enumerate(counts):
                flattened.append([step_idx+1, count, 0])

        df = pd.DataFrame(data=np.array(flattened),
                          columns=['chord_position', 'num_chord_changes', 'unique'])

        sns.plt.clf()

        sns.set(style="whitegrid")
        g = sns.factorplot(x='chord_position', y='num_chord_changes', hue='unique', data=df,
                           size=6, kind="bar", palette="muted")
        g.despine(left=True)
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(PLOT_DIR, '%s.pdf' % plot_id))

    def plot_stats_by_step(self, num_changes, num_unique_changes,
                           plot_id):
        print 'num_changes', num_changes
        print 'num_unique_changes', num_unique_changes
        flattened = []
        for step_idx, counts in num_changes.iteritems():
            for j, count in enumerate(counts):
                # 0 being not unique
                # 1 being unique
                flattened.append([step_idx+1, count, 0])
                flattened.append([step_idx+1, num_unique_changes[step_idx][j], 1])

        df = pd.DataFrame(data=np.array(flattened),
                          columns=['chord_position', 'num_chord_changes', 'unique'])

        sns.plt.clf()

        sns.set(style="whitegrid")
        g = sns.factorplot(x='chord_position', y='num_chord_changes', hue='unique', data=df,
                           size=6, kind="bar", palette="muted", ci=None)
        g.despine(left=True)
        plt.legend(loc='upper right')
        # g.set_ylabels("survival probability")

        # sns.set(style="ticks")
        # sns.boxplot(x='chord_position', y='num_chord_changes', hue='unique',
        #             data=df, palette="PRGn")
        # sns.despine(offset=10, trim=True)
        # plt.tight_layout()

        plt.savefig(os.path.join(PLOT_DIR, '%s.pdf' % plot_id))














