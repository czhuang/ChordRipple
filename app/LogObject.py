

import time

from music21_chord_tools import letter2music21, is_roman_numeral


def get_timestamp():
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    return timestamp


class LogObject(object):
    def __init__(self, kind, seq, tags, rating=None):
        self.kind = kind
        self.seq = self.parse_seq(seq)
        self.tags = tags
        self._rating = rating

        self.timestamp = get_timestamp()

    def __repr__(self):
        if self.rating is not None:
            line = '%s, %s, %d' % (self.kind, self.seq, self.rating)
        else:
            line = '%s, %s' % (self.kind, self.seq)

        if self.tags is not None:
            line += ', tags: '
            for k in self.tags.keys():
                line += '%s, ' % k

        if self.tags is not None and 'suggestion_item' in self.tags:
            item = self.tags['suggestion_item']
            if item is not None:
                line += 'inds: '
                for ind in self.tags['suggestion_item'].inds:
                    line += '%d, ' % ind
        return line

    @staticmethod
    def parse_seq(seq):
        if isinstance(seq, list):
            return seq
        seq_raw = seq.strip().split(' ')
        seq_list = []
        for s in seq_raw:
            if len(s) > 0:
                if not is_roman_numeral(s):
                    s = letter2music21(s)
                seq_list.append(s)
        return seq_list

    @property
    def rating(self):
        return self._rating

    @rating.setter
    def rating(self, value):
        self._rating = value

    def match_tags(self, tags):
        if not isinstance(self.tags, dict):
            return False
        matches_all = True
        for key, val in tags.iteritems():
            if key not in self.tags or val != self.tags[key]:
                matches_all = False
                break
        return matches_all
