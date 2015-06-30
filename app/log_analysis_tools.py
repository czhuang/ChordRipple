import os

import cPickle as pickle

import numpy as np

import pylab as plt

from Logs import LOG_FOLDER

from diversity_tools import compute_set_diversity
from retrieve_model_tools import retrieve_SkipGramNN


class LogAnalyzer():
    def __init__(self, logs):
        self.logs = logs
        self.model = retrieve_SkipGramNN()
        # for sym in self.model.syms:
        #     print sym

    @staticmethod
    def unique_syms(syms):
        unique_syms = []
        for sym in syms:
            if sym not in unique_syms:
                unique_syms.append(sym)
        return unique_syms

    def log_diversity(self, kind='use'):
        if kind == 'use':
            logs = self.logs.use
        else:
            logs = getattr(self.logs, kind)

        # assume the last one is the final
        logs_subsets = [[logs[0]], logs[1:-1], [logs[-1]]]
        logs_syms = []
        for log_subset in logs_subsets:
            subset_syms = []
            for log in log_subset:
                subset_syms.extend(log.seq)
            subset_syms = self.unique_syms(subset_syms)
            logs_syms.append(subset_syms)

        diversity_scores = [compute_set_diversity(syms, self.model)
                            for syms in logs_syms]
        print diversity_scores

        print 'in intermediate but no in final...'
        syms_not_in_final = []
        for sym in logs_syms[1]:
            if sym not in logs_syms[2]:
                syms_not_in_final.append(sym)
        print len(syms_not_in_final), syms_not_in_final
        print "final_syms", logs_syms[-1]

    def plot_type_against_time(self):
        # TODO: need to distinguish play user or machine
        # types = self.logs.get_all_types()
        # y-axis ordering according to degree of exploration
        type_ordering = ['save', 'edit', 'play', 'use', 'play', 'play bold']
        type_ordering_str = ['save', 'edit', 'play (edit)', 'use', 'play (suggestions)', 'play bold (suggestions)']
        ys = []
        for log in self.logs:
            if log.kind in type_ordering:
                ind = type_ordering.index(log.kind)
            else:
                print 'WARNING: not yet considering this type', log.kind
                continue

            if 'play' in log.kind:
                if log.tags['author'] == 'user':
                    ind = 2
                elif log.tags['author'] == 'machine':
                    ind = 4
                else:
                    assert False, 'WARNING: case not considered'
            ys.append(ind)

        plt.plot(ys, 'x-')
        plt.ylim([0, len(type_ordering_str)-1])
        plt.yticks(range(len(type_ordering_str)-1), type_ordering_str)

        plt.title('sequence of interactions (condition: %s)' % self.logs.experiment_type)
        plt.xlabel('interaction step index')
        plt.ylabel('interaction type')
        plt.savefig('seq_of_interactions-%s.pdf' % self.logs.id)


def load_test_log():
    fname = 'exampleLog'

    fpath = os.path.join(LOG_FOLDER, fname)
    with open(fpath, 'rb') as p:
        logs = pickle.load(p)

    logs.reprocess_seqs()
    return logs


def analyze_temporal_actions():
    logs = load_test_log()

    log_analysis = LogAnalyzer(logs)
    log_analysis.plot_type_against_time()


def check_logs():
    logs = load_test_log()
    print logs
    # print logs.compute_save_rating_summary()
    filtered_logs = logs.use_sub_ripple
    print '...use ripple substitutions'
    print '# of instances: ', len(filtered_logs)
    for log in filtered_logs:
        print log

    filtered_logs = logs.played_machine
    print '...play ripple'
    print '# of instances: ', len(filtered_logs)
    for log in filtered_logs:
        print log

    print logs.compute_use_adventurousness()


def plot_bar_graph():
    x = np.arange(5)
    y1, y2 = np.random.randint(1, 25, size=(2, 5))
    width = 0.25
    plt.figure()
    plt.bar(x, y1, width)
    plt.bar(x+width, y2, width, color=plt.rcParams['axes.color_cycle'][2])
    ax = plt.gca()
    ax.set_xticks(x+width)
    ax.set_xticklabels(['a', 'b', 'c', 'd', 'e'])
    plt.savefig('bar_plot.pdf')

if __name__ == '__main__':
    # analyze_temporal_actions()
    plot_bar_graph()
