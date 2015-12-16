
import os
import glob

import numpy as np


def print_vector(arr, name, truncate=True):  #, num_decimals=5):
    arr = np.array(arr)
    print '%s: (%.3f, %.5f)' % (name, np.max(arr), np.min(arr)),
    # print '%s: (%.3f, %.5f)' % (name, np.max(arr), np.min(arr)),
    n_items = np.minimum(5, arr.size)
    if not truncate:
        n_items = arr.size
    for val in arr[:n_items]:
        # print "%.5f" % val,
        print "%.2f" % val,
    print


def print_array(arr, name):
    if not isinstance(arr, np.ndarray) and not isinstance(arr, list):
        arr = arr.value
    arr = np.atleast_2d(arr)
    n_rows = np.minimum(5, arr.shape[0])
    n_cols = np.minimum(9, arr.shape[1])
    print "\n----", name, arr.shape

    if arr.size > 1:
        col_max = np.max(arr, axis=0)
        print_vector(col_max, 'max')
        col_min = np.min(arr, axis=0)
        print_vector(col_min, 'min')

    for i in range(n_rows):
        for j in range(n_cols):
            print "%.3f" % arr[i, j],
        print


def subseqs_to_text(subseq, seqs, other_info=None, fname_tag=''):
    print len(seqs)
    for seq in seqs:
        print seq
    lines = ''
    for i, seq in enumerate(seqs):
        if other_info is not None:
            lines += '%s, ' % other_info[i]
        for s in seq:
            lines += '%s, ' % s
        lines += '\n'
    subseq_str = '_'.join(subseq).replace('/', '_')
    fname = 'subseqs-%s-%s.txt' % (subseq_str, fname_tag)
    with open(fname, 'w') as p:
        p.writelines(lines)


def retrieve_most_recent_fname(fpath, fext='.pkl'):
    try:
        fname = max(glob.iglob(os.path.join(fpath, '*'+fext)), key=os.path.getctime)
    except ValueError:
        fname = None
    print 'fname:', fname
    return fname
