
from copy import copy

import numpy as np
import pylab as plt
from pylab import setp
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

from music_theory_tools import RELATIVE_MINOR, CIRCLE_OF_FIFTHS_MINOR_DICT, CIRCLE_OF_FIFTHS_MAJOR_DICT
from load_songs_tools import get_raw_data, get_configs


def plot_bach_dist():
    configs = get_configs()

    print configs
    configs['corpus'] = 'bach'
    configs['min_count'] = 1

    from make_model_tools import make_Ngram
    ngram = make_Ngram(configs)
    counts, syms = ngram.get_sorted_counts()

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(counts)), counts)
    plt.title(' Sorted chord counts in Bach chorale corpus', fontsize=16)
    plt.ylabel('Chord counts', fontsize=14)
    plt.xlabel('Chords indexed by descending counts',  fontsize=14)

    setp(plt.gca().get_xticklabels(), fontsize=12)
    setp(plt.gca().get_xticklabels(), fontsize=10)

    plt.savefig('bach-counts.pdf')



def plot_bar_sorted_counts():
    pass



def plot_mat(mat, title, syms, y_tick_syms=None):
    mat = np.asarray(mat)
    assert mat.size == len(syms)
    from colormaps import inferno
    plt.matshow(mat, cmap=inferno)
    # plt.title(title)
    from pylab import setp
    # if mat.shape[1] < 23:
    #     fontsize = 'medium'
    # else:
    #     fontsize = 'xx-small'
    fontsize = 'small'
    fontsize = 'x-small'
    if y_tick_syms is None:
        y_tick_syms = syms
    #print '# of syms:', len(syms)
    print ' '.join(syms)
    if len(y_tick_syms) == 1:
        if y_tick_syms[0] == 'V':
            y_tick_str = '$'+y_tick_syms[0]+'_1$'
        else:
            y_tick_str = '$'+y_tick_syms[0]+'$'
        plt.yticks([0], [y_tick_str], fontweight='bold')
        setp(plt.gca().get_yticklabels(), fontsize='large')
    else:
        plt.yticks(range(len(y_tick_syms)), y_tick_syms, fontweight='bold')
        setp(plt.gca().get_yticklabels(), fontsize=fontsize)


    plt.xticks(range(len(syms)), syms, fontweight='bold')

    setp(plt.gca().get_xticklabels(), fontsize=fontsize)
    plt.title(title)
    # plt.colorbar(shrink=.8)
    colorbar = plt.colorbar(shrink=.7, ticks=np.arange(-2.0, 2.1, 1.0))
    if len(y_tick_syms) == 1:
        colorbar.set_label('chord weights in '+y_tick_str, fontsize=9)  # labelpad=-40, y=0.45)
    colorbar.ax.tick_params(labelsize=9)
    # plt.tight_layout()


def plot_mat_pca_ordered(bigram, data, configs,
                         fname_tag='', unigram=None):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    pca.fit(bigram)
    transformed_bigram = np.squeeze(pca.transform(bigram))
    print transformed_bigram.shape
    print 'variance covered: %.2f' % np.sum(pca.explained_variance_ratio_)
    plot_mat_sort_with(transformed_bigram, data, configs,
                       bigram, unigram, fname_tag=fname_tag, technique='PCA')
    return transformed_bigram


def plot_mat_sort_with(values_to_sort_with, data, configs,
                       bigram, unigram=None, fname_tag='', technique='PCA'):
    sorted_inds = np.argsort(values_to_sort_with)
    print sorted_inds

    syms = data.syms
    sorted_syms = [syms[ind] for ind in sorted_inds]
    # sorted_ngram = bigram[sorted_inds]
    print bigram.shape
    sorted_ngram = [bigram[ind, sorted_inds] for ind in sorted_inds]
    sorted_ngram = np.squeeze(np.asarray(sorted_ngram))
    print sorted_ngram.shape
    plt.clf()
    title_str = '%s sorted %s transition matrix' % (technique, fname_tag)
    plot_mat(sorted_ngram, title_str, sorted_syms)
    plt.savefig('all-%s-ordered-%s-%s.pdf' %
                (technique, fname_tag, configs.name))

    if unigram is None:
        return
    sorted_unigram = [unigram[ind] for ind in sorted_inds]
    line = ''
    for i in range(len(syms)):
        line += '\n%s (%d): ' % (sorted_syms[i], sorted_unigram[i])
        local_sorted_inds = np.argsort(sorted_ngram[i, :])[::-1]
        local_sorted_syms = [sorted_syms[ind] for ind in local_sorted_inds]
        for j in range(len(syms)):
            line += '%s (%.2f), ' % (local_sorted_syms[j],
                                     sorted_ngram[i, local_sorted_inds[j]])
    fname = 'transitions-%s-bach.txt' % fname_tag
    with open(fname, 'w') as p:
        p.writelines(line)


def plot_mat_sorted_with_itself(values_to_sort_with, syms,
                                configs, row_tag, topn=None, save=False,
                                title_str='', fname_tag=''):
    #print 'shape', values_to_sort_with.shape
    sorted_inds = np.argsort(-values_to_sort_with)
    sorted_inds = np.squeeze(sorted_inds)
    # syms = data.syms
    if topn is None:
        topn = len(syms)
    sorted_syms = [syms[ind] for ind in sorted_inds[:topn]]
    sorted_ngram = np.sort(values_to_sort_with)[::-1][:topn]

    if save:
        plt.clf()
    #print sorted_ngram.shape
    if len(sorted_ngram.shape) == 1:
        sorted_ngram = sorted_ngram[None, :]
    #print sorted_ngram.shape
    plot_mat(sorted_ngram, title_str,
             sorted_syms, y_tick_syms=[row_tag])

    if save:
        plt.savefig('trans-%s-%s.pdf' % (fname_tag, configs.name))


def project_3d_to_2d(xyz, ax):
    x2, y2, _ = proj3d.proj_transform(xyz[0], xyz[1], xyz[2], ax.get_proj())
    xy = np.array([x2, y2])
    return xy


# def annotate(syms, vecs, pl_ax, is_3d, color='b', text_size=None):
#     print '# of syms:', len(syms)
#     print vecs.shape
#     for i, sym in enumerate(syms):
#         xy = vecs[i, :]
#         if is_3d:
#             xy = project_3d_to_2d(xy, pl_ax)
#         # if DUPLICATE_BY_ROTATE:
#         #     text_size = 'small'
#         # else:
#         #     text_size = 'xx-small'
#         if text_size is None:
#             # text_size = 'large'
#             text_size = 'xx-small'
#         pl_ax.annotate(sym, xy=xy, xytext=(-3, 2),
#                        textcoords = 'offset points', size=text_size, color=color)


def annotate(syms, vecs, pl_ax, text_size=None, color='b'):
    assert len(syms) == vecs.shape[0]
    print '# of syms:', len(syms)
    print vecs.shape
    for i, sym in enumerate(syms):
        xy = vecs[i, :]
        if text_size is None:
            text_size = 'x-small'
            text_size = 'small'
        if text_size == 'x-large':
            fontweight = 'bold'
        else:
            fontweight = 'normal'
        pl_ax.annotate(sym, xy=xy, xytext=(-3, 2),
                       textcoords = 'offset points', size=text_size, color=color, fontweight=fontweight)


def add_arrow_annotation(syms, vecs, arrow_dict, pl_ax, is_3d=False, color='#ee8d18',
                         linewidth=3):
    if not is_3d:
        assert vecs.shape[1] == 2
    else:
        assert vecs.shape[1] == 3

    highlight_syms = arrow_dict.keys()
    filtered_vecs, filtered_syms = filter_syms(vecs, syms, include_syms=highlight_syms)
    print highlight_syms
    print filtered_syms
    # assert len(highlight_syms) == len(filtered_syms)
    annotate(filtered_syms, filtered_vecs, pl_ax, color=color, text_size='x-large')

    for start, end in arrow_dict.iteritems():
        if start not in syms or end not in syms:
            continue
        start_ind = syms.index(start)
        end_ind = syms.index(end)
        start_pos = vecs[start_ind, :]
        if is_3d:
            start_pos = project_3d_to_2d(start_pos, pl_ax)
        # end_pos = (vecs[end_ind, :] - vecs[start_ind, :])
        end_pos = vecs[end_ind, :]
        if is_3d:
            end_pos = project_3d_to_2d(end_pos, pl_ax)
        diff = end_pos - start_pos
        # head_length = np.sqrt(np.square(end_pos[0]) + np.square(end_pos[1]))*0.1
        pl_ax.arrow(start_pos[0], start_pos[1], diff[0], diff[1],
                    fc=color, ec=color, head_width=0, head_length=0,
                    linewidth=linewidth)  # head_width=0.05, head_length=head_length,




def add_most_annotations(syms, vecs, ax, is_3d, plot_relative_minor=True):
    # annotate(syms, vecs, ax, is_3d)
    # add_arrow_annotation(syms, vecs, second_res_dict, ax, is_3d)
    add_arrow_annotation(syms, vecs, CIRCLE_OF_FIFTHS_MAJOR_DICT, ax, is_3d, color='g')
    add_arrow_annotation(syms, vecs, CIRCLE_OF_FIFTHS_MINOR_DICT, ax, is_3d, color='m')
    if plot_relative_minor:
        add_arrow_annotation(syms, vecs, RELATIVE_MINOR, ax, is_3d)


def add_secondary_dominants_annotations(syms, vecs, second_res_dict, ax, is_3d):
    annotate(syms, vecs, ax, is_3d)
    add_arrow_annotation(syms, vecs, second_res_dict, ax, is_3d)


def add_relative_minor_annotations(syms, vecs, ax, is_3d):
    annotate(syms, vecs, ax, is_3d)
    add_arrow_annotation(syms, vecs, RELATIVE_MINOR, ax, is_3d)


def make_song_dict(song):
    song_dict = {}
    for i in range(len(song)-1):
        song_dict[song[i]] = song[i+1]
    return song_dict


def plot_vec(vecs, syms, configs, highlight_syms=[],
             with_annotations=False, save=False,
             fname_tag=None, return_ax=False,
             zero_gram_vec=None, second_res_dict=None, plot3d=False,
             subplot=False, doPCA=False):

    # TODO: hacky settings
    PLOT_SECOND_ONLY = False
    PLOT_RELATIVE_MINOR_ONLY = False
    PLOT_RELATIVE_MINOR = False
    ONLY_SHOW_HIGHLIGH_SYMBOLS = False

    original_vecs = vecs
    from postprocessing_tools import pca_project
    if doPCA:
        if plot3d is True:
            vecs = pca_project(vecs, n_components=3)
        else:
            vecs = pca_project(vecs, n_components=2)
    else:
        if not plot3d:
            assert vecs.shape[1] == 2
        else:
            assert vecs.shape[1] == 3

    if zero_gram_vec is None:
        dot_sizes = 2  # 5
    else:
        dot_sizes = np.exp(np.log(zero_gram_vec))/60
    highlight_dot_sizes = 25

    if vecs.shape[1] < 3:
        plot3d = False

    if PLOT_SECOND_ONLY:
        print 'second_res_dict', len(second_res_dict)
        print second_res_dict

    if len(highlight_syms) == 0:
        from music_theory_tools import CIRCLE_OF_FIFTHS_MAJOR, CIRCLE_OF_FIFTHS_MINOR
        highlight_syms.extend(CIRCLE_OF_FIFTHS_MAJOR[::])
        highlight_syms.extend(CIRCLE_OF_FIFTHS_MINOR[::])

    highlight_vecs = []
    highlight_syms_copy = copy(highlight_syms)
    for s in highlight_syms_copy:
        if s in syms:
            ind = syms.index(s)
            highlight_vecs.append(vecs[ind, :])
        else:
            print 'WARNING: %s is not in syms' % s
            highlight_syms.remove(s)
    highlight_vecs = np.squeeze(highlight_vecs)

    if plot3d:
        print '3D'
        if not subplot:
            fig = plt.figure()
        # TODO: fig might not be defined for 3D subplot
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(vecs[:, 0], vecs[:, 1], vecs[:, 2],
                   s=highlight_dot_sizes, color='b')

        if len(highlight_vecs) > 0:
            ax.scatter(highlight_vecs[:, 0], highlight_vecs[:, 1],
                       highlight_vecs[:, 2], s=dot_sizes, color='r')

        if not with_annotations:
            annotate(syms, vecs, ax, True)
        elif PLOT_SECOND_ONLY:
            add_secondary_dominants_annotations(syms, vecs, second_res_dict, ax, True)
        elif PLOT_RELATIVE_MINOR_ONLY:
            add_relative_minor_annotations(syms, vecs, ax, True)
        else:
            add_most_annotations(syms, vecs, ax, True,
                                 plot_relative_minor=PLOT_RELATIVE_MINOR)
    else:
        print '2D', np.shape(vecs)
        if not subplot:
            plt.figure()
        if vecs.shape[1] > 1:
            if not ONLY_SHOW_HIGHLIGH_SYMBOLS:
                plt.scatter(vecs[:, 0], vecs[:, 1], s=dot_sizes, color='b')

            if len(highlight_vecs) > 0:
                plt.scatter(highlight_vecs[:, 0], highlight_vecs[:, 1],
                            s=highlight_dot_sizes, color='k')

            ax = plt.gca()
            if len(highlight_syms) == 0 or not ONLY_SHOW_HIGHLIGH_SYMBOLS:
                annotate(syms, vecs, ax, False)
            else:
                annotate(highlight_syms, highlight_vecs, ax, False)

            if not with_annotations:
                pass
            elif PLOT_SECOND_ONLY:
                add_secondary_dominants_annotations(syms, vecs, second_res_dict, ax)
            elif PLOT_RELATIVE_MINOR_ONLY:
                add_relative_minor_annotations(syms, vecs, ax, False)
            else:
                add_most_annotations(syms, vecs, ax, False, plot_relative_minor=PLOT_RELATIVE_MINOR)
        else:
            # just 1D
            plt.plot(vecs, 'x', )
            # pl_ax.annotate(sym, xy=xy, xytext=(-3, 2), textcoords = 'offset points', size=text_size, color='b')

    # title_str = ' '.join(SIMPLE_CHORDS_ORDERED)
    # plt.title('chord space')
    plt.tick_params(axis='both', which='major', labelsize=6)

    fname = configs.name
    # fname = get_fname_base(vecs, syms, seqs, vec_dim, window_sz=window_sz)
    # fname = get_fname_base(vecs, syms, seqs, min_count=min_count, vec_dim=vec_dim,
    #                        do_pca=doPCA, window_sz=window_sz)
    if plot3d:
        fname += '-3d'
    if PLOT_SECOND_ONLY:
        fname += '-second'
    if PLOT_RELATIVE_MINOR_ONLY or PLOT_RELATIVE_MINOR:
        fname += '-relative_minor'
    if fname_tag is not None:
        fname += '-%s' % fname_tag
    if save:
        plt.savefig('%s.pdf' % fname)

    if return_ax:
        # TODO: ax might not be defined depending on usage
        return fname, ax, vecs
    else:
        return fname


def filter_syms(vecs, syms, exclude_syms=None, include_syms=None):
    filtered_vecs = []
    filtered_syms = []
    if exclude_syms is not None:
        for i, sym in enumerate(syms):
            if sym not in exclude_syms:
                filtered_vecs.append(vecs[i, :])
                filtered_syms.append(sym)
    else:
        filtered_vecs = vecs.copy()
        filtered_syms = copy(syms)

    filtered_vecs = np.asarray(filtered_vecs)

    filtered_again_vecs = []
    filtered_again_syms = []
    if include_syms is not None:
        for i, sym in enumerate(filtered_syms):
            if sym in include_syms:
                filtered_again_vecs.append(filtered_vecs[i, :])
                filtered_again_syms.append(sym)
    else:
        filtered_again_vecs = filtered_vecs.copy()
        filtered_again_syms = copy(filtered_syms)
    return np.asarray(filtered_again_vecs), filtered_again_syms


if __name__ == '__main__':
    plot_bach_dist()