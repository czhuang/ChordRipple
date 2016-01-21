
import os
from collections import OrderedDict
from copy import copy

TEST_FLAG = True
TEST_PATH = 'rs_harmony_test'
ALL_PATH = 'rs200_harmony'


def get_rock_corpus_fpaths():
    if TEST_FLAG:
        path = TEST_PATH
    else:
        path = ALL_PATH

    fnames = os.listdir(path)
    annotator = 'tdc'

    annotator_fnames = []
    for fname in fnames:
        if '_' + annotator in fname:
            annotator_fnames.append(os.path.join(path, fname))
    return annotator_fnames


# TODO: make bar parser to work out duration...
# TODO: make bar class...
# TODO: get test file that has lines that start with $BP
# TODO: and has comment at the end of line...



def parse_chords(fpath):
    print 'fpath: ', fpath
    with open(fpath, 'r') as p:
        lines = p.readlines()

    # remove empty lines
    lines = [line.strip() for line in lines if len(line.strip()) > 0]

    comments = OrderedDict()
    sections = OrderedDict()
    file_content_ordering = []
    song_ordering = []
    key_sig = None
    time_sig = None
    for i, line in enumerate(lines):
        # if comments start from the beginning and take up the whole line
        if line[0] == '%':
            label = 'c%d' % len(comments)
            comments[label] = line
            file_content_ordering.append(label)
        # if song-structure line that has optional key and timesig
        elif line.startswith('S:'):
            items = line.split()
            for item in items[1:]:
                item = item.strip()
                if '[' not in item and '/' not in item:
                    song_ordering.append(item[1:])
                elif '[' in item and '/' not in item:  # key signature
                    key_sig = item[1:-1]
                elif '[' in item:  # time signature
                    time_sig = item[1:-1]
                else:
                    assert False, "Case should not exist..."
        else:
            if ':' not in line:
                print line
                assert False, "Guessed incorrectly the kind of line"
            print 'line', line
            # some lines don't have bar lines at all
            if '|' in line:
                bars = line.split('|')
                # if '|' was the last character, it marks the end
                # but split adds another entry after it, so we're removing that
                if len(bars[-1].strip()) == 0:
                    bars = bars[:-1]
            else:
                bars = [line]
            label_first_bar = bars[0].split(':')
            section_label = label_first_bar[0]
            bars[0] = label_first_bar[1]
            bar_events = []
            for bar in bars:
                bar = bar.strip()
                if len(bar) == 0:
                    # TODO: need to record that it was a reptition
                    # TODO: need to make a class for chordEvent to add that attribute
                    bar_events.append(bar_events[-1])
                    continue

                items = bar.strip().split()

                # if repeat previous bar
                items[0] = items[0].strip()
                if items[0].startswith('*'):
                    num_repeats = int(items[0][1:])
                    for _ in range(num_repeats):
                        bar_events.append(bar_events[-1])
                    items = items[1:]
                events = []
                for j, item in enumerate(items):
                    item = item.strip()

                    # if reached the comments at the end
                    if item[0] == '%':
                        label = 'c%d' % len(comments)
                        comments[label] = ' '.join(items[j:])
                        file_content_ordering.append(label)
                        break

                    if item[0] == '$' and '*' in item:
                        parts = item.split('*')
                        # the part after * gives the # of repetitions
                        for _ in range(int(parts[1])):
                            # TODO: assumes that ref symbol phrases always take up a bar
                            bar_events.append([parts[0]])
                    elif item[0] == '$':
                        bar_events.append([item])
                    else:
                        events.append(item)
                if len(events) > 0:
                    bar_events.append(events)

            sections[section_label] = bar_events
            file_content_ordering.append(section_label)

    if key_sig is None:
        assert False, 'Key signature not specified.  In C?'
    if time_sig is None:
        time_sig = '4/4'

    print file_content_ordering
    print song_ordering
    print key_sig, time_sig

    for key, items in comments.iteritems():
        print key, items

    for key, items in sections.iteritems():
        print key, items

    song = TDC_Song(sections, comments, key_sig, time_sig)
    return song


class Chord(object):
    def __init__(self, name, dur, held_from_before):
        self._name = name
        self._dur = dur
        self.held_from_before = held_from_before


class Rule(object):
    def __init__(self, label, bars, rules, time_sig):
        self._name = label
        self._bars = bars
        self.rules_ref = rules
        self.time_sig = time_sig

        self._expanded_bars = self.expand_bars()
        self._expanded_chords = self.expand_to_chords()
        self._expanded_chord_strs = self.make_chord_strs()

    def expand_bars(self):
        # still list of bar lists
        expanded_bars = []
        for bar in self._bars:
            if bar[0] == '$':
                label = bar[0][1:]
                expanded_bars.extend(copy(self.rules_ref[label]))
            else:
                expanded_bars.append(bar)
        return expanded_bars

    def expand_to_chords(self):
        expanded_chords = []
        for bar in self._expanded_bars:
            expanded_bar, time_sig = self.expand_bar_to_chords(bar, time_sig)
            expanded_chords.extend(expanded_bar)
        return expanded_chords

    def expand_bar_to_chords(self, bars, time_sig):
        chords = []
        for bar in bars:
            if '/' in bar[0]:
                time_sig = bar[1:-1]
                bar = bar[1:]

            num_beats = time_sig.split('/')[0]

            # for a bar with more than 2 item will most likely have '.'
            durs = []
            if len(bar) <= 2 and num_beats == 4:
                dur_unit = num_beats / len(bars)
                assert num_beats % len(bars) == 0
                durs = [dur_unit] * len(bars)

            dur_previous = None
            for i, ch in enumerate(bar):
                if len(durs) > 0:
                    dur = durs[i]
                elif i + 1 < len
                ch = Chord(ch, dur_unit, hel)
                chords.append(ch)
            else:
                for ch in bar:

    def make_chord_strs(self):




class TDC_Song(object):
    def __init__(self, sections, comments, key_sig, time_sig):
        self._sections = sections
        self._comments = comments
        self._key_sig = key_sig
        self._time_sig = time_sig

    def get_section_str(self):
        time_sig = self._time_sig
        section_strs = {}
        for key, bars in self._sections.iteritems():
            section_str = ''




def main():
    fpaths = get_rock_corpus_fpaths()
    song = parse_chords(fpaths[0])
    song.get_section_str()


if __name__ == '__main__':
    main()