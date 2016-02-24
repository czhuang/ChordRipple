import numpy as np
import matplotlib.pyplot as plt


uses = [(2, 36), (14, 22), (4, 18), (0, 32), (8, 16), (10, 20), (2, 18), (6, 18), (6, 16)]

N = len(uses)

ripple_counts = [use[0] for use in uses]
# in ripple condition
use_counts = [use[1] for use in uses]

sorted_inds = np.argsort(use_counts)
sorted_use_counts = [use_counts[ind] for ind in sorted_inds]
sorted_ripple_counts = [ripple_counts[ind] for ind in sorted_inds]
sorted_singleton_counts = [sorted_use_counts[i] - sorted_ripple_counts[i] for i in range(N)]

ind = np.arange(N)    # the x locations for the groups
width = 0.45 #0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, sorted_ripple_counts, width, color='r')#, yerr=menStd)
p2 = plt.bar(ind, sorted_singleton_counts, width, color='y',
             bottom=sorted_ripple_counts)#, yerr=womenStd)

plt.ylabel('Total use counts')
plt.title('Proportion of ripple uses')

xtick_labels = ['P%d' % (i+1) for i in sorted_inds]
plt.xlim([-width, N-width/4])
plt.xticks(ind + width/2., xtick_labels)

plt.yticks(np.arange(0, np.max(use_counts)+1, 5))
plt.legend((p1[0], p2[0]), ('ripple', 'singleton'), loc=2)

plt.savefig('rippleProportion.png')