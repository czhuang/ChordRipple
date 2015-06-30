
import os

import cPickle as pickle

from Logs import Logs
from Logs import LOG_FOLDER
from log_analysis_tools import load_test_log


def check_logs():
    logs = load_test_log()
    print logs

    print '\n\n======== use ========='
    uses = logs.use
    for use in uses:
        print use
        # print use.seq
    print '=================='

    print '\n\n======== changes ========='
    uses = logs.changes
    for use in uses:
        print use
    print '=================='

    # print logs.compute_save_rating_summary()
    filtered_logs = logs.use_sub_ripple
    print '\n\n...use ripple substitutions'
    print '# of instances: ', len(filtered_logs)
    for log in filtered_logs:
        print log

    filtered_logs = logs.played_machine
    print '\n\n...play ripple'
    print '# of instances: ', len(filtered_logs)
    for log in filtered_logs:
        print log

    print '\n\n======= experiment type:', logs.experiment_type
    print
    print logs.compute_use_adventurousness()


if __name__ == '__main__':
    check_logs()