
import os
from collections import OrderedDict
from copy import copy

import cPickle as pickle

TEST_FLAG = False
TEST_PATH = 'rs_harmony_test'
ALL_PATH = 'rs200_harmony'

# GLOBAL_COUNT_UNPARSABLE_SEQS = 0


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
            song_ordering = []
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
            sections['S'] = song_ordering
            file_content_ordering.append('S')
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


class Rule(object):
    def __init__(self, label, bars, rules, time_sig):
        self._name = label
        self._compact_bars = bars
        self._rules_ref = rules
        self._time_sig = time_sig
        self._referenced_labels = self.collect_referenced_labels()
        self._dur_unit = self.get_smallest_dur()

    def expand(self):
        # TODO: hack, consider 'S' rule case too
        # the parse of 'S' is currently missing the '$'
        # S ['Ln', 'Vr1', 'Ch', 'Ln*2', 'Ch', 'Vr2', 'Ch*2', 'Ou']
        if self._name != 'S':
            print '---expanding', self._name
            self._bars = self.expand_to_phrases()
            self._chords = self.expand_to_chords()
            self._beats, self._beats_w_onsets = self.expand_to_beats()

    def get_smallest_unit(self):


    def collect_referenced_labels(self):
        labels = set()
        for i, bar in enumerate(self._compact_bars):
            if len(bar) > 0 and bar[0][0] == '$':
                label = bar[0][1:]
                if label not in labels:
                    labels.add(label)
        return labels

    @property
    def referenced_labels(self):
        return self._referenced_labels

    @property
    def bars(self):
        return self._bars

    @property
    def beats(self):
        return self._beats
    
    @property
    def beats_w_onsets(self):
        return self._beats_w_onsets
    
    def expand_to_phrases(self):
        # for example if "Vr1: [4/4] I . . IV | $A I . . V | $B I . . IV | $A $B"
        # expand out the rules
        # still list of bar lists
        #print 'expand_to_phrases:', self._compact_bars
        expanded_bars = []
        beginning_with_rest = True
        for i, bar in enumerate(self._compact_bars):
            if not len(bar) == 1 or bar[0] != 'R':
                beginning_with_rest = False
            if bar[0][0] == '$':
                label = bar[0][1:]
                # print 'label to expand', label
                expanded_bars.extend(copy(self._rules_ref[label].bars))
            elif not beginning_with_rest:
                expanded_bars.append(bar)
        return expanded_bars

    def expand_to_chords(self):
        #print 'expand_to_chords:', self._bars
        expanded_chords = [None]
        time_sig = self._time_sig
        for bars in self._bars:
            expanded_bar, new_time_sig = self.expand_bar_to_chords(bars, time_sig, expanded_chords[-1])
            if expanded_bar is not None:
                expanded_chords.extend(expanded_bar)
                time_sig = new_time_sig
        expanded_chords.pop(0)
        return expanded_chords

    def is_time_sig(self, string):
        # [4/4]
        time_sig = string[1:-1]
        try:
            pos = string.index('/')
        except ValueError:
            return False, time_sig
        if not string[1:pos].isdigit() or not string[pos+1:len(string)-1].isdigit():
            return False, time_sig
        return True, time_sig

    def is_key_change(self, string):
        if string[0] != '[' or string[-1] != ']' or '/' in string:
            return False, None
        key = string[1:-1]
        for ch in key:
            if ch.isdigit():
                return False, key
        return True, key

    def expand_bar_to_chords(self, bar, time_sig, last_chord):
        # TODO: assumes the time signature is in the position
        # print 'expand_bar_to_chords:', time_sig, bar
        chords = [last_chord]
        beat_pos = 1

        num_events = len(bar)
        for i, ch in enumerate(bar):
            is_time_sig, temp_time_sig = self.is_time_sig(ch)
            if is_time_sig:
                # TODO: assumes that there is only one time signature change
                num_events -= 1
                # print time_sig
                continue
            is_key_change, temp_key = self.is_key_change(ch)
            # print ch, is_key_change
            # TODO: right now just ignoring bars with key change
            if is_key_change:
                return None, None

            # TODO: maybe dur_unit should be proportion of bar not actual beat duration
            num_beats = int(time_sig.split('/')[0])
            dur_unit = num_beats // num_events

            # TODO: right now ignoring the bars that don't divide up perfectly
            if num_beats % num_events != 0:
                # GLOBAL_COUNT_UNPARSABLE_SEQS += 1
                return None, None

            # assert num_beats % num_events == 0
            if ch == '[0]':
                ch = 'R'
            if ch != '.':
                ch = Chord(ch, beat_pos, dur_unit)
                chords.append(ch)
            else:
                chords[-1].add_dur(dur_unit)
                # chords[-1].add_dur(1)

            beat_pos += dur_unit
        chords.pop(0)
        return chords, time_sig

    def expand_to_beats(self):
        beats = []
        beats_w_onsets = []
        for ch in self._chords:
            assert isinstance(ch.dur, int)
            for i in range(ch.dur):
                beats.append(ch.name)
                if i == 0:
                    beats_w_onsets.append((ch.name, 1))
                else:
                    beats_w_onsets.append((ch.name, 0))
        return beats, beats_w_onsets


class Chord(object):
    def __init__(self, name, onset_beat_pos, dur):
        self._name = name
        self._onset_beat = onset_beat_pos
        self._dur = dur

    @property
    def name(self):
        return self._name

    @property
    def dur(self):
        return self._dur

    def add_dur(self, dur):
        self._dur += dur


class TDC_Song(object):
    def __init__(self, sections, comments, key_sig, time_sig):
        self._sections = sections
        self._comments = comments
        self._key_sig = key_sig
        self._time_sig = time_sig
        
        self._rules = self.make_rules()
        
    def make_rules(self):
        rules = OrderedDict()
        for left, right in self._sections.iteritems():
            rules[left] = Rule(left, right, rules, self._time_sig)

        # need to check the order
        keys = rules.keys()
        print 'keys ordering:', keys
        for i, label in enumerate(keys):
            referenced_labels = rules[label].referenced_labels
            all_in = True
            for ref_label in referenced_labels:
                if ref_label not in keys[:i+1]:
                    all_in = False
                    break
            if not all_in:
                # this puts this entry in the back of the dict
                temp = rules[label]
                del rules[label]
                rules[label] = temp
                print 'reordered keys:', rules.keys()

        for rule in rules.values():
            rule.expand()
        return rules

    def get_section_beats(self):
        return [', '.join(rule.beats) + '\n' for label, rule in self._rules.iteritems() if len(label) >= 2]
    
    def get_section_beats_w_onsets(self):
        return [rule.beats_w_onsets for label, rule in self._rules.iteritems() if len(label) >= 2]
        

def save_as_text(fpath, seqs):
    seqs_strs = []
    for seq in seqs:
        seq_str = ', '.join(seq) + '\n'
        seqs_strs.append(seq_str)
    fpath = os.path.splitext(fpath)[0] + '.txt'
    print 'save fname:', fpath
    with open(fpath, 'w') as p:
        p.writelines(seqs_strs)


def main():
    fpaths = get_rock_corpus_fpaths()
    seqs = []
    seqs_w_onsets = []
    for fpath in fpaths:
        print '--------------', fpath
        song = parse_chords(fpath)
        sections = song.get_section_beats()

        print '--- sections ---'
        for sec in sections:
            print sec

        seqs.extend(sections)
        seqs_w_onsets.extend(song.get_section_beats_w_onsets())

    print '# of segmented seqs:', len(seqs)
    # fpath = os.path.join('data', 'rock-rns-phrases.txt')
    fpath = os.path.join('data', 'rock-rns-phrases.txt')
    with open(fpath, 'w') as p:
        p.writelines(seqs)
    fpath = os.path.join('data', 'rock-rns-phrases.pkl')
    output = {'seqs': seqs, 'seqs_w_onsets': seqs_w_onsets}
    with open(fpath, 'wb') as p:
        pickle.dump(output, p)
    print 'pickled fname:', fpath

    # print 'GLOBAL_COUNT_UNPARSABLE_SEQS', GLOBAL_COUNT_UNPARSABLE_SEQS


if __name__ == '__main__':
    main()