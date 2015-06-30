
import os

import cPickle as pickle

import numpy as np

from LogObject import LogObject, get_timestamp


LOG_FOLDER = 'logs'


class Logs(list):
    def __init__(self, experiemnt_type, experiment_type_strs):
        list.__init__(self)
        self.id = get_timestamp()
        self.snapshot_count = 0

        self._experiment_type = experiemnt_type
        self.experiment_type_strs = experiment_type_strs

    def __repr__(self):
        if hasattr(self, 'experiment_type'):
            lines = 'Experiment type: %s' % self.experiment_type
        else:
            lines = 'Experment type not recorded yet'
        lines += 'Logs: (%d)\n' % len(self)
        for lg in self:
            lines += '%s\n' % lg
        return lines

    def reprocess_seqs(self):
        for log in self:
            log.seq = LogObject.parse_seq(log.seq)

    @property
    def experiment_type(self):
        return self.experiment_type_strs[self._experiment_type]

    @property
    def snapshot_name(self):
        name = '%s--%s' % (self.id, self.snapshot_count)
        self.snapshot_count += 1
        return name

    def dump(self):
        fname = self.snapshot_name
        fpath = os.path.join(LOG_FOLDER, fname)
        with open(fpath, 'wb') as p:
            pickle.dump(self, p)

    def add(self, kind, text, tags=None):
        obj = LogObject(kind, text, tags)
        self.append(obj)
        print 'added:', obj
        print '--- total # items:', len(self)
        if kind == 'save' or kind == 'use':
            self.dump()

    def get_types(self, kinds):
        filtered_logs = []
        for log in self:
            for kind in kinds:
                if kind in log.kind:
                    filtered_logs.append(log)
                    break
        return filtered_logs

    def get_types_tags(self, kinds, tags):
        filtered_logs = []
        for log in self:
            for i, kind in enumerate(kinds):
                # if log.kind == kind and log.match_tags(tags[i]):
                if kind in log.kind and log.match_tags(tags[i]):
                    filtered_logs.append(log)
        return filtered_logs

    @property
    def use(self):
        # author: computer
        return self.get_types('use')

    @property
    def use_sub_ripple(self):
        logs = self.use
        filtered_logs = []
        for log in logs:
            if log.tags is not None and 'suggestion_item' in log.tags:
                item = log.tags['suggestion_item']
                if item is None:
                    print 'WARNING: somehow use item is None'
                    print log.tags['author']

                elif item.kind == 'sub_ripple':
                    filtered_logs.append(log)
        return filtered_logs

    @property
    def play(self):
        # author: could be computer or user
        # and need to distinguish if played and not chosen [not preferred]
        # vs playing back input textbox [preferred]
        return self.get_types('play')

    @property
    def played_machine(self):
        return self.get_types_tags(['play'], [{'author': 'machine'}])

    @property
    def played_user(self):
        return self.get_types_tags(['play'], [{'author': 'machine'}])

    @property
    def save(self):
        # author: user (since save only from input textbox)
        return self.get_types(['save'])

    @property
    def edit(self):
        # user edit changes
        return self.get_types(['edit'])

    @property
    def changes(self):
        # include both user edit and used changes
        return self.get_types(['edit', 'use'])

    def add_rating(self, rating, type_, text, ind):
        logs = self.save
        print '...add_rating'
        for log in logs:
            print log
        print 'add_rating', logs[ind], text
        assert logs[ind] != text
        logs[ind].rating = rating

    def get_all_types(self):
        types = []
        for log in self:
            if log.kind not in types:
                types.append(log.kind)
        return types

    def compute_save_rating_summary(self):
        logs = self.save
        ratings = [log.rating for log in logs if log.rating is not None]
        print 'ratings', ratings
        return np.mean(ratings), np.std(ratings)

    def compute_use_adventurousness(self):
        print '... comupute_adventurousness'
        logs = self.use_sub_ripple
        context_ranks = []
        used_ripple = []

        for log in logs:
            if 'suggestion_item' in log.tags:
                item = log.tags['suggestion_item']
                item_dict = item.tags
                if item_dict is not None:
                    print item_dict, item.seq, item.inds
                if item_dict is not None and 'context_rank' in item_dict:
                    context_ranks.append(item_dict['context_rank'])
                    if len(item.inds) > 1:
                        used_ripple.append(1.0)
                    else:
                        used_ripple.append(0.0)
        # print context_ranks, np.mean(context_ranks), np.std(context_ranks)
        # print used_ripple

        used_adventurous = [ind for ind in range(len(context_ranks))
                            if context_ranks[ind] > 5]

        # out of > 5 how mnay used ripple, < 5 how many used ripple
        used_ripple_adventurous = [used_ripple[ind] for ind in range(len(context_ranks))
                                   if context_ranks[ind] > 5]
        num_used_ripple_adventurous = np.sum(used_ripple_adventurous)

        # print 'used_ripple_adventurous', used_ripple_adventurous
        # used_ripple_typical = [ used_ripple[ind] for ind in range(len(context_ranks)) \
        #                             if context_ranks[ind] <= 5 ]
        # print 'used_ripple_typical', used_ripple_typical
        #
        # print 'used_ripple_adventurous', np.mean(used_ripple_adventurous)
        # print 'used_ripple_typical', np.mean(used_ripple_typical)

        self.context_ranks = context_ranks

        num_used = len(logs)
        num_ripple = np.sum(used_ripple)

        result = {}
        result['num_used'] = len(logs)
        result['num_ripples'] = np.sum(used_ripple)
        result['num_adventurous'] = len(used_adventurous)
        result['used_ripple_adventurous'] = np.sum(used_ripple_adventurous)

        for k, v in result.iteritems():
            print k, v
        return result
