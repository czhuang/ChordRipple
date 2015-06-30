
import os

from copy import deepcopy
import cPickle as pickle

import numpy as np

from music21 import pitch

from socketio.namespace import BaseNamespace
from socketio.mixins import BroadcastMixin

from dynamic_programming_tools import shortest_path, simple_foward_backward_gap_dist
from retrieve_model_tools import retrieve_NGram, retrieve_SkipGramNN
from music21_chord_tools import sym2chord, roman2letter, is_roman_numeral, letter2roman
# from music21_chord_tools import sym2roman
# from diversity_tools import get_diverse_seqs
from config import get_configs

from Logs import Logs


PKL_FOLDER = 'data'

EXPERIMENT_TYPE_STRS = ['manual', 'baseline_singleton', 'ripple']
MANUAL, BASELINE_SINGLETON, RIPPLE = range(len(EXPERIMENT_TYPE_STRS))
EXPERIMENT_TYPE = RIPPLE

from music21_chord_tools import ROMAN2LETTER_RELABELS
from music21_chord_tools import ROMAN_PARTIAL_RELABELS
from music21_chord_tools import MAKE_NOTE_EXCEPTIONS
from music21_chord_tools import LETTER2ROMAN_RELABELS
from music21_chord_tools import LETTER_PARTIAL_RELABELS_FOR_USER


class SuggestionList(object):
    def __init__(self, parent):
        self.items = list()
        self.types = self.make_suggestion_types()
        self.parent = parent

    @property
    def num_items(self):
        return len(self.items)

    def retrieve_item_at(self, ind, text):
        print 'retrieve_item_at', ind, 'total items', len(self.items)
        if ind >= len(self.items):
            return None
        item = self.items[ind]
        retrieve_text = ' '.join(item.seq)
        print text.strip()
        print retrieve_text
        assert text.strip() == retrieve_text
        return item

    def add(self, items):
        if not isinstance(items, list):
            items = [items]
        for item in items:
            _, item.seq = self.parent.format_seq(item.seq)

            if not self.in_list(item):
                self.items.append(item)

    def add_seqs_inds(self, seqs, inds, kind, tags=None):
        for i, seq in enumerate(seqs):
            self.items.append(SuggestionItem(seq, inds[i], kind, tags))

    def in_list(self, item):
        in_list_ = False
        for check_item in self.items:
            if item.seq == check_item.seq:
                in_list_ = True
                break
        return in_list_

    def get_seqs_inds(self):
        seqs = [item.seq for item in self.items]
        inds = [item.inds for item in self.items]
        return seqs, inds

    def clear(self):
        del self.items[:]

    def make_suggestion_types(self):
        type_texts = {}
        type_texts['next'] = "What's next?"
        type_texts['sub_ripple'] = "Substitutions for the current chord, and its ripples"
        type_texts['till_end'] = "Bringing us to the end of a phrase"
        return type_texts


class SuggestionItem(object):
    def __init__(self, seq, inds, kind, tags=None):
        self.seq = seq
        self.inds = inds
        self.kind = kind
        self.tags = tags


class Resource(BaseNamespace, BroadcastMixin):
    def initialize(self):
        # Called after socketio has initialized the namespace.
        self.history = []
        self.parsed_seqs_notes = {}
        self.unit_dur = 60/92.0

        self.ngram = retrieve_NGram()
        self.nn = retrieve_SkipGramNN()

        assert self.ngram.syms == self.nn.syms

        self.previous_sym = None
        self.previous_sym_ind = None
        self.n_suggestions = 5
        self.n_similar = 2

        self.suggestions = SuggestionList(self)
        self.suggestions_above = SuggestionList(self)

        self.config = get_configs()
        self.corpus = self.config['corpus']
        print '...corpus', self.corpus

        if self.config['use_letternames']:
            self.symbol_type = 'letter'
        else:
            self.symbol_type = 'roman'

        # need to correct some roman numerals
        print '# of syms: %d' % len(self.ngram.syms)
        self.syms = []
        for sym in self.ngram.syms:
            formatted_sym, valid = self.format_sym(sym)
            self.syms.append(formatted_sym)

        # print 'F#m in syms?', 'F#m' in self.syms

        # need to update the "spelling" of roman numerals in nn and ngram
        self.nn.syms = self.syms
        self.ngram.syms = self.syms

        self._rn2letter, self._letter2rn = self.load_rn2letter_dict()

        self.experiment_type = EXPERIMENT_TYPE

        self.logs = Logs(EXPERIMENT_TYPE, EXPERIMENT_TYPE_STRS)


    @property
    def model(self):
        return self.ngram

    def load_rn2letter_dict(self):
        fname = 'rn2letter-rock.pkl'
        fpath = os.path.join(PKL_FOLDER, fname)
        with open(fpath, 'rb') as p:
            rn2letter = pickle.load(p)
        letter2rn = {}

        # TODO: is this a bug? should be letter2rn[val] = key
        for key, val in rn2letter.iteritems():
            letter2rn[key] = val
        return rn2letter, letter2rn

    def on_inputSave(self, text):
        # log add automatically saves upon seeing type "save"
        self.logs.add("save", text)

    def on_clear(self, text):
        self.logs.add("clear", text)
        self.clear_suggestions()

    def on_rating(self, value, ind, text):
        try:
            ind = int(ind)
        except ValueError:
            print 'WARNING: index to save entry wanted, but received', ind

        if len(text) > 0:
            self.logs.add_rating(value, 'save', text, ind)
        logs = self.logs.save
        for log in logs:
            print log

    def on_setPlaybackSpeed(self, speed):
        print '...playbackspeed',
        self.unit_dur = 60.0 / speed
        print self.unit_dur

    def disconnect(self, *a, **kw):
        super(Resource, self).disconnect(*a, **kw)

    def on_ping(self, param):
        print '---on_ping', param
        self.emit('pong', param)

    def on_requestData(self):
        # data = [{'x':10, 'y':15, 'label':'I'},
        #         {'x':10, 'y':15, 'label':'IV'}]
        fname = 'w1-RN.pkl'
        with open(fname, 'rb') as p:
            vertices = pickle.load(p)
            labels = pickle.load(p)

        # vertices = [[100, 150], [200, 150]]
        # labels = ['I', 'IV']
        max_x = np.max(vertices[:, 0])
        min_x = np.min(vertices[:, 0])
        x_range = max_x - min_x

        max_y = np.max(vertices[:, 1])
        min_y = np.min(vertices[:, 1])
        y_range = max_y - min_y

        width = 750.0
        margin = 30
        scale = (width-2*margin)/y_range
        if x_range > y_range:
            scale = (width-2*margin)/x_range

        vertices[:, 0] -= min_x
        vertices[:, 1] -= min_y
        vertices *= scale
        vertices[:, 0] += margin
        vertices[:, 1] += margin

        vertices = map(list, vertices)
        self.emit('dataLabels', vertices, labels)

    def parse_seq_as_notes(self, text):
        text = text.strip()
        if text in self.parsed_seqs_notes.keys():
            chords = self.parsed_seqs_notes[text]
            return chords
        parts, raw_parts = self.parse_seq_as_syms(text)
        chords = self.make_note_seqs(parts)
        self.parsed_seqs_notes[text] = chords
        return chords

    def parse_seq_as_syms(self, text):
        text = text.strip()
        parts = text.split(' ')
        parts = [part for part in parts if len(part) > 0]
        parts, raw_parts = self.format_seq(parts)
        return parts, raw_parts

    def make_note_seqs(self, seq):
        print 'make_note_seqs', seq
        note_seq = []
        for sym in seq:
            notes = self.make_notes(sym)
            if len(note_seq) == 0:
                note_seq.append(notes)
                continue
            if len(notes) == 0:
                note_seq.append([])
                continue
            if len(note_seq[-1]) == 0:
                diff = 0
            else:
                diff = np.min(notes) - np.min(note_seq[-1])
            # print '...make_note_seqs', sym, notes, diff, diff % 12
            if np.abs(diff) > 6:
                # i.e. C4 to B4 => C4 to B3
                # i.e. C4 to B5 => C4 to B3
                shift_int = (np.abs(diff) / 12) * 12
                # prefer going down
                # TODO: temp disable
                if np.abs(diff) % 12 > 6 and diff > 0:
                    shift_int += 12
                    # shift_int -= 12

                direction = np.sign(diff) * -1
                notes = np.asarray(notes) + shift_int * direction
                print 'notes[0]', notes[0]
                if notes[0] <= 55:
                    notes += 12
                note_seq.append(list(notes))
            else:
                note_seq.append(notes)
            # print 'shifted', notes

        print 'playSeq, chords'
        for i, ch in enumerate(note_seq):
            print seq[i],
            for p in ch:
                print pitch.Pitch(p).nameWithOctave,
            print ch
        print

        return note_seq

    def make_notes(self, sym):
        sym = sym.strip()
        if sym in MAKE_NOTE_EXCEPTIONS:
            # look up in dictionary
            letter = self.rn2letter(sym)
            print 'due to roman not giving right pitches', sym, letter
            sym = letter
        chord_sym = sym2chord(sym)
        midi = []
        if chord_sym is not None:
            midi = [pch.midi for pch in chord_sym.pitches]
            # take away duplicates
            midi = list(set(midi))
            midi = sorted(midi)
            # double the tonic on top
            if sym in ['I', 'I7', 'i']:
                doubled_note = midi[0]
                midi.append(doubled_note+12)
                print 'doubled tonic on top', midi
            elif len(midi) > 4 and '11' in sym:
                # 11th creates half step with 3rd if major (not checking if major here)
                # shift 11th up an octave to make
                print midi
                # 1,3,11,5,7,9
                reduced_midi = midi[:2] + [midi[3]] + [midi[2]+12]
                midi = reduced_midi
            elif len(midi) > 4:
                reduced_midi = midi[:3] + [midi[-1]]
                midi = reduced_midi
        return midi

    def make_log_tags_for_playback(self, text, ind, author, suggestPanel):
        tags = {}
        print 'make_log_tags_for_playback', author
        if author == 'machine':
            item = self.retrieve_suggestion_item_at(ind, text, suggestPanel)
            if item is None:
                print 'WARNNG: can not retrieve suggestion item', ind, text
                assert False
            tags['suggestion_item'] = item
        tags['author'] = author
        return tags

    def on_playSeq(self, text, pos, author, item_ind, suggestPanel=None, loop=False):
        print '...on_playSeq, loop', loop
        if author == 'dont_log':
            pass
        elif loop:
            tags = self.make_log_tags_for_playback(text, item_ind, author, suggestPanel)
            self.logs.add("loop", text, tags)
        else:
            tags = self.make_log_tags_for_playback(text, item_ind, author, suggestPanel)
            self.logs.add("play", text, tags)

        print '---on_parseSeq:', text
        chords = self.parse_seq_as_notes(text)
        durs = [self.unit_dur] * len(chords)
        if loop:
            chords = chords[:] + chords[:]
            durs = durs[:-1] + [durs[-1]+self.unit_dur*0.2] + durs[:]

        # durs = [self.unit_dur] * len(chords)
        print chords
        print durs

        self.emit('playSeq', chords, durs)

    def on_playSubseq(self, chord_changes, author, item_ind, suggestPanel, log=True):
        print '...on_playSubseq', chord_changes, author, item_ind, log, suggestPanel
        print chord_changes
        syms = []
        inds = []
        # assuming continuous
        for i, chord_change in enumerate(chord_changes):
            if chord_change[1]:
                syms.append(chord_change[0])
                inds.append(i)
        print len(inds), inds
        if len(inds) > 1:
            p_ind = inds[0]
            for ind in inds[1:]:
                if ind - p_ind != 1:
                    print 'WARNING: Changes not continuous'
                p_ind = ind

        # TODO: a bit of a hack here
        all_chords = [ch[0] for ch in chord_changes]

        # add to log
        text = ' '.join(all_chords)
        if log:
            tags = self.make_log_tags_for_playback(text, item_ind, author, suggestPanel)
            self.logs.add("play bold", text, tags)

        # need context to determine octave
        all_notes = self.parse_seq_as_notes(text)
        notes = [all_notes[ind] for ind in inds]
        durs = [self.unit_dur] * len(syms)
        self.emit('playSeq', notes, durs)

    def get_similar_chords(self, sym, topn):
        similars = self.nn.most_similar(sym, topn=topn)
        if similars is None:
            return
        sims = [s[0] for s in similars]
        return sims

    def on_startSeqs(self):
        if self.experiment_type == MANUAL:
            return
        # want diverse sequences
        print '---on_startSeqs'
        # TODO: fixed start seqs
        if self.symbol_type == 'roman':
            # sims = [[u'I', u'vi'], ['V', 'vi'], ['IV', 'ii'], [u'V7/IV', u'IV'], [u'i', u'i'], [u'i', u'V6']]
            sims = [[u'i'], [u'I'], ['V'], ['iv'], ['IV'], ['V7/IV']]
        else:
            # sims = [['a', 'C64'], ['G', 'C'], ['A', 'gb'], ['C', 'F'], ['G', 'e'], ['Bb', 'Ab'], ['E', 'G']]
            # sims = [['C'], ['F'], ['G'], ['Am'], ['Cm'], ['B-'], ['A-']]
            sims = [['Cm'], ['B-'], ['F/C'], ['G7'], ['Dm7'], ['Cmaj7'], ['F#dim']]

        self.clear_suggestions()
        for s in sims:
            self.suggestions.add(SuggestionItem(s, range(len(s)), 'start'))
        seqs, inds = self.suggestions.get_seqs_inds()
        # print seqs
        # print inds
        self.emit('updateChordSuggestions', seqs, inds)

    def rn2letter(self, sym):
        if sym in ROMAN2LETTER_RELABELS:
            formatted_sym = ROMAN2LETTER_RELABELS[sym]
        elif sym in self._rn2letter:
            formatted_sym = self._rn2letter[sym]
            print 'rn2letter retrieved', sym, formatted_sym
        else:
            formatted_sym = roman2letter(sym)
        return formatted_sym

    def letter(self, sym):
        if is_roman_numeral(sym):
            return self.rn2letter(sym)
        else:
            return sym

    def roman(self, sym):
        if sym is None:
            return None
        if is_roman_numeral(sym):
            return sym
        else:
            return self.letter2rn(sym)

    def letter2rn(self, sym):
        if sym in LETTER2ROMAN_RELABELS:
            formatted_sym = LETTER2ROMAN_RELABELS[sym]
        elif sym in self._letter2rn:
            formatted_sym = self._letter2rn[sym]
            print 'letter2rn retrieved', formatted_sym
        else:
            formatted_sym = letter2roman(sym)
            if formatted_sym is None:
                return ''
            # print 'created', formatted_sym

        return formatted_sym

    def back_to_roman(self, sym):
        is_roman = is_roman_numeral(sym)
        if not is_roman:
            sym = self.letter2rn(sym)
        return sym

    def format_seq(self, seq):
        local_seq = []
        local_original_seq = []
        for sym in seq:
            formatted_sym, valid = self.format_sym(sym)
            if valid:
                local_seq.append(formatted_sym)
            else:
                local_seq.append('')
            local_original_seq.append(formatted_sym)

        return local_seq, local_original_seq

    def format_sym(self, sym):
        is_roman = is_roman_numeral(sym)
        formatted_sym = sym
        # print 'format_sym', sym, is_roman

        if is_roman and self.symbol_type != 'roman':
            formatted_sym = self.rn2letter(sym)
        elif not is_roman and self.symbol_type == 'roman':
            formatted_sym = self.letter2rn(sym)

        if formatted_sym is not None:
            if self.symbol_type == 'roman':
                for k, v in ROMAN_PARTIAL_RELABELS.iteritems():
                    if k in formatted_sym:
                        formatted_sym = formatted_sym.replace(k, v)
            else:
                for k, v in LETTER_PARTIAL_RELABELS_FOR_USER.iteritems():
                    if k in formatted_sym:
                        formatted_sym = formatted_sym.replace(k, v)

        # check = sym2chord(sym)
        # if check is None:
        #     return None
        if formatted_sym == '':
            return sym, False
        return formatted_sym, True

    def generate_subs_from_context(self, sym_ind, original_seq, factor=1):
        print '...generate_subs_from_context', sym_ind, original_seq
        # factor: factor*n_similar # of suggestions
        original_sym = original_seq[sym_ind]
        subs = []
        # sorted_syms = None
        # if 0 < sym_ind < len(original_seq):
        if sym_ind - 1 < 0:
            before_sym = None
        else:
            before_sym = original_seq[sym_ind - 1]
            if before_sym not in self.syms:
                before_sym = None
        if sym_ind + 1 >= len(original_seq):
            after_sym = None
        else:
            after_sym = original_seq[sym_ind + 1]
            if after_sym not in self.syms:
                after_sym = None
        sorted_probs, sorted_syms = \
            simple_foward_backward_gap_dist(self.model, before_sym, after_sym)
        n_subs = factor*self.n_similar
        if sorted_syms is not None:
            subs = sorted_syms[:n_subs]
        if original_sym in subs:
            subs.remove(original_sym)
            subs.append(sorted_syms[n_subs])
        print '...subs', subs
        return sorted_syms, subs

    def make_single_sub_suggestion_items(self, sym_ind, original_seq, subs, sorted_syms, return_tags_only=False):
        # original_sym = original_seq[sym_ind]
        suggestion_items = []
        tags_list = []
        for i, ss in enumerate(subs):
            sub_seq = original_seq[:sym_ind] + [ss] + original_seq[sym_ind + 1:]
            print 'subseq', sub_seq
            tags = {}
            if i < self.n_similar:
                tags['source'] = 'subs'
            else:
                tags['source'] = 'sim'

            if sorted_syms is not None:
                tags['context_rank'] = sorted_syms.index(ss)
            tags_list.append(tags)
            item = SuggestionItem(sub_seq, [sym_ind], 'sub_ripple', tags)
            suggestion_items.append(item)

        # original_sym, valid = self.format_sym(original_sym)
        # if valid:
        #     subs.insert(0, original_sym)
        if return_tags_only:
            return tags_list
        return suggestion_items

    def generate_singleton_subs(self, sym_ind, original_seq, factor=1):
        original_sym = original_seq[sym_ind]
        print '...generate substitutions based on similarity'
        # generate substitutions based on similarity
        sims = self.get_similar_chords(original_sym, self.n_similar)
        print sims
        # generate substitutions based on context
        sorted_syms, subs = self.generate_subs_from_context(sym_ind, original_seq, factor=factor)
        # collect all the single changes
        if subs is None:
            subs = sims
        elif sims is not None:
            subs.extend(sims)
        # subs first, sims next
        return sorted_syms, subs

    def on_generateAlternatives(self, text, pos):
        if self.experiment_type == MANUAL:
            return
        print '--- --- generate_alternative --- ---', text, pos
        print self.symbol_type
        original_seq, raw_original_seq = self.parse_seq_as_syms(text)
        print 'original_seq', original_seq
        if len(original_seq) == 0 or original_seq == ['']:
            self.previous_sym = None
            self.previous_sym_ind = None
            self.clear_suggestions()
            self.emit('updateChordSuggestions', [], [], [])
            self.on_startSeqs()
            return

        original_sym, sym_ind = self.get_sym_at_pos(text, pos, return_ind=True)
        print original_sym, sym_ind

        if sym_ind is None or original_seq[sym_ind] != original_sym:
            print 'WARNING: symbol not matching'
            return
        print 'formatted sym', original_sym, sym_ind
        print 'previous', self.previous_sym, self.previous_sym_ind

        # self.roman(self.previous_sym) == self.roman(original_sym)
        if self.previous_sym is not None and self.previous_sym_ind == sym_ind \
                and self.previous_sym == original_sym:
            return
        else:
            self.previous_sym = original_sym
            self.previous_sym_ind = sym_ind
            self.clear_suggestions()

        if original_sym is None or not len(original_sym):
            print 'WARNING: no symbol at this position'
            # if sym in middle, then use context to ripple in to generate suggestions
            sorted_syms, subs = self.generate_subs_from_context(sym_ind, raw_original_seq, 4)
            if sorted_syms is not None:
                suggestion_items = self.make_single_sub_suggestion_items(sym_ind, raw_original_seq, subs, sorted_syms)
                self.clear_suggestions()
                self.suggestions_above.add(suggestion_items)
                seqs, inds = self.suggestions_above.get_seqs_inds()

                print '...generateAlternatives, # of items', self.suggestions_above.num_items
                self.emit('updateChordSuggestionsAbove', seqs, inds, self.suggestions_above.types)
            return

        # generate nexts
        print '...generate_nexts', raw_original_seq
        print original_seq, original_sym
        if sym_ind == len(original_seq) - 1:
            ss, sinds = self.generate_next(original_sym, sym_ind, raw_original_seq)
            if ss is None:
                print 'WARNING: no next chords for ', original_seq[sym_ind]
                return
            for i, s in enumerate(ss):
                self.suggestions.add(SuggestionItem(s, sinds[i], 'next'))
                print s, sinds[i]

        if self.experiment_type == RIPPLE:
            suggestion_items = self.generate_side_ripples(sym_ind, original_seq)
            self.suggestions.add(suggestion_items)

        sorted_syms, subs = self.generate_singleton_subs(sym_ind, raw_original_seq)
        suggestion_items = self.make_single_sub_suggestion_items(sym_ind, raw_original_seq, subs, sorted_syms)

        if subs is None:
            seqs, inds = self.suggestions_above.get_seqs_inds()
            self.emit('updateChordSuggestionsAbove', seqs, inds)
            return

        # generate ripples for the single changes
        print '...subs', subs
        if self.experiment_type == RIPPLE:
            seq_subs, seq_inds = self.generate_ripples(raw_original_seq, sym_ind, subs)
        else:
            seq_subs = None

        if seq_subs is None:
            # add what we have so far
            self.suggestions_above.add(suggestion_items)
        else:
            # the first one is for the current text
            # check if it is the same as current text
            # same_as_original_seq = True
            # ss = seq_subs.pop(0)
            # inds = seq_inds.pop(0)
            # for s, ind in zip(ss, inds):
            #     if len(original_seq) < ind and original_seq[ind] != s:
            #         same_as_original_seq = False
            # rippled_plus_original_items = []
            # if not same_as_original_seq:
            #     tags = {'source': 'user'}
            #     rippled_plus_original_items.append(SuggestionItem(ss, inds, 'sub_ripple', tags))

            rippled_plus_original_items = []
            assert len(suggestion_items) == len(seq_subs)
            # interleave the two
            print '...interleaving subs and their ripples'
            for i, item in enumerate(suggestion_items):
                rippled_plus_original_items.append(item)
                ripple_item = SuggestionItem(seq_subs[i], seq_inds[i], 'sub_ripple', item.tags)
                rippled_plus_original_items.append(ripple_item)
            self.suggestions_above.add(rippled_plus_original_items)

        # if EXPERIMENT_TYPE != BASELINE_SINGLETON and sym_ind == len(original_seq) - 1:
        #     ss, sinds = self.generate_continuations(original_sym, sym_ind, original_seq)
        #     self.suggestions_above.add_seqs_inds(ss, sinds, 'till_end')

        seqs, inds = self.suggestions.get_seqs_inds()
        print '...generateAlternatives, # of items', self.suggestions.num_items
        self.emit('updateChordSuggestions', seqs, inds, self.suggestions.types)

        seqs, inds = self.suggestions_above.get_seqs_inds()
        print '...generateAlternatives, # of above items', self.suggestions_above.num_items
        print seqs
        print inds
        self.emit('updateChordSuggestionsAbove', seqs, inds, self.suggestions.types)

    def generate_continuations(self, sym, ind, original_seq):
        postfix_len = 4
        seqs = []
        seq_inds = []
        for i in range(2, postfix_len):
            fixed = {ind:sym}
            fixed[ind+i] = 'I'
            seq, inds = \
                shortest_path(self.model, fixed, ind, original_seq)
            seqs.append(seq)
            seq_inds.append(inds)
        return seqs, seq_inds

    def generate_next(self, sym, seq_ind, original_seq):
        trans = self.model.trans
        syms = self.syms
        if sym not in syms:
            return None, None
        sym_ind = syms.index(sym)
        n_conts = self.n_suggestions
        inds = np.argsort(-trans[sym_ind, :])[:n_conts]
        # unformatted_syms = [syms[ind] for ind in inds]
        # print 'unformatted_syms', unformatted_syms
        # formatted_subs = []
        # for ind in inds:
        #     formatted_sub = self.format_sym(syms[ind])
        #     print syms[ind], formatted_sub
        #     if formatted_sub is not None:
        #         formatted_subs.append(ind)
        # print len(inds), len(formatted_subs)
        # subs = [original_seq[:] + [formatted_subs[i]] for i in range(len(inds))]

        subs = [original_seq[:] + [syms[ind]] for ind in inds]
        # print 'generate_next', subs
        return subs, [[seq_ind+1]]*n_conts

    def generate_ripples(self, original_seq, sym_ind, sims, win_max=2):
        print '...generate_ripples', sims
        seq_subs = []
        seq_inds = []
        for win in range(1, win_max):
            ub = sym_ind + win
            lb = sym_ind - win

            # allow one extra seq step
            lb_out_bound = lb < -1
            ub_out_bound = ub > len(original_seq)

            # supposedly already second time out of bound
            if lb_out_bound or ub_out_bound:
                break

            if lb < 0:
                lb = 0
            if ub > len(original_seq):
                ub = len(original_seq)

            for j, s in enumerate(sims):
                fixed = {}
                if ub < len(original_seq):
                    fixed[ub] = original_seq[ub]
                else:
                    fixed[ub] = []

                fixed[lb] = original_seq[lb]
                # may override lb or ub
                fixed[sym_ind] = s
                print fixed

                sub_seq, sym_inds = \
                    shortest_path(self.model, fixed, sym_ind, original_seq)

                seq_subs.append(sub_seq)
                seq_inds.append(list(sym_inds))
        for seq in seq_subs:
            print seq
        return seq_subs, seq_inds

    def generate_side_ripples(self, sym_ind, original_seq, factor=1, win_max=2):
        print '...generate_side_ripples', sym_ind, 'len(original_seq)', len(original_seq)
        # because if factor not equal to one for now will cause the sub, sim attribution to be incorrect
        # even though the context rank will probably reveal which one it is, and is more important anyway
        assert factor == 1
        # original_sym = original_seq[sym_ind]
        # left side
        left_subs = None
        if sym_ind > 0:
            left_ind = sym_ind - 1
            left_sorted_syms, left_subs = self.generate_singleton_subs(left_ind, original_seq, factor)

        # right side
        right_subs = None
        if sym_ind < len(original_seq) - 1:
            right_ind = sym_ind + 1
            right_sorted_syms, right_subs = self.generate_singleton_subs(right_ind, original_seq, factor)

        if left_subs is None and right_subs is None:
            print 'no side ripple yet'
            return []

        print 'left subs', left_subs
        print 'right subs', right_subs
        seqs = []
        inds = []

        n = 0
        # choose the smaller non-zero one
        if left_subs is None:
            left_n = 0
        else:
            left_n = len(left_subs)
            n = left_n
        if right_subs is None:
            right_n = 0
        else:
            right_n = len(right_subs)
            n = right_n

        if right_n > left_n > 0:
            n = left_n

        print left_n, right_n, n
        for i in range(n):
            if left_subs is not None and right_subs is not None:
                seq = original_seq[:left_ind] + [left_subs[i]] + \
                    [original_seq[sym_ind]] + [right_subs[i]]
                if right_ind + 1 < len(original_seq):
                    seq += original_seq[right_ind+1:]
                inds.append([left_ind, right_ind])

            elif left_subs is None:
                seq = original_seq[:right_ind] + [right_subs[i]]
                if right_ind + 1 < len(original_seq):
                    seq += original_seq[right_ind+1:]
                inds.append([right_ind])

            elif right_subs is None:
                seq = original_seq[:left_ind] + [left_subs[i]] + \
                    original_seq[sym_ind:]
                inds.append([left_ind])
            else:
                assert False, 'ERROR: case not considered'

            seqs.append(seq)

        tags = None
        if left_subs is not None:
            tags = self.make_single_sub_suggestion_items(left_ind, original_seq,
                                                         left_subs, left_sorted_syms,
                                                         return_tags_only=True)
        right_tags = None
        if right_subs is not None:
            right_tags = self.make_single_sub_suggestion_items(right_ind, original_seq,
                                                               right_subs, right_sorted_syms,
                                                               return_tags_only=True)

        # merge the tags
        if tags is not None and right_tags is not None:
            for i in range(n):
                key = 'context_rank'
                tag = tags[i]
                other_tag = right_tags[i]
                if key in tags and key in other_tag:
                    tag[key] = [tag[key], other_tag[key]]
                elif key in other_tag:
                    tag[key] = [other_tag[key]]
                print 'tags'
        elif tags is None:
            tags = right_tags

        suggestion_items = []
        for i, seq in enumerate(seqs):
            print 'seq:', seq
            print 'inds', inds[i]
            print tags[i]
            item = SuggestionItem(seq, inds[i], 'side_ripple', tags[i])
            suggestion_items.append(item)

        return suggestion_items

    # TODO: more side options
    def generate_sides(self, original_seq, sym_ind, sims, win_max=2):
        seq_subs = []
        seq_inds = []
        for win in range(1, win_max):
            ub = sym_ind + win
            lb = sym_ind - win

            # allow one extra seq step
            lb_out_bound = lb < -1
            ub_out_bound = ub > len(original_seq)

            # supposedly already second time out of bound
            if lb_out_bound or ub_out_bound:
                break

            if lb < 0:
                lb = 0
            if ub > len(original_seq):
                ub = len(original_seq)

            for j, s in enumerate(sims):
                fixed = {}
                if ub < len(original_seq):
                    fixed[ub] = original_seq[ub]
                else:
                    fixed[ub] = []

                fixed[lb] = original_seq[lb]
                # may override lb or ub
                fixed[sym_ind] = s
                print fixed

                sub_seq, sym_inds = \
                    shortest_path(self.model, fixed, sym_ind, original_seq)

                seq_subs.append(sub_seq)
                seq_inds.append(list(sym_inds))
        return seq_subs, seq_inds

    def generate_more_change_alternatives(self, text, pos):
        sym, sym_ind = self.get_sym_at_pos(text, pos, return_ind=True)
        if sym is None:
            return
        original_seq, raw_original_seq = self.parse_seq_as_syms(text)
        win_max = 2

        sims = self.get_similar_chords(sym, 3)
        sims.insert(0, sym)
        if sims is None:
            return
        return self.generate_ripples(raw_original_seq, sym_ind, sims, win_max)

    def clear_suggestions(self):
        self.previous_sym = None
        self.previous_sym_ind = None
        self.suggestions.clear()
        self.suggestions_above.clear()
        self.emit('updateChordSuggestions', [], [], [])
        self.emit('updateChordSuggestionsAbove', [], [], [])
        return

    def retrieve_suggestion_item_at(self, ind, text, suggestPanel):
        if 'above' in suggestPanel:
            item = self.suggestions_above.retrieve_item_at(ind, text.strip())
        else:
            item = self.suggestions.retrieve_item_at(ind, text.strip())
        return item

    def on_textChange(self, text, pos, kind, author, ind, suggestPanel=None, play=True):
        print '--- on_textChange ---', text, pos, author, play, ind, suggestPanel
        if len(text.strip()) == 0:
            return
        tags = {}
        if kind == 'use' and author == 'machine':
            assert suggestPanel is not None
            item = self.retrieve_suggestion_item_at(ind, text, suggestPanel)
            tags['suggestion_item'] = item
            tags['suggestion_list'] = deepcopy(self.suggestions.items)
        elif kind == 'use' and author == 'user':
            self.clear_suggestions()

        tags['author'] = author
        self.logs.add(kind, text, tags)
        self.history.append(text)
        if text is None or not isinstance(text, unicode):
            print "WARNING: chord symbol not valid", text
            return
        sym, sym_ind = self.get_sym_before_pos(text, pos, return_ind=True)
        print 'sym to play?', sym, sym_ind, play
        note_seqs = self.parse_seq_as_notes(text)
        print 'len(note_seqs)', len(note_seqs)
        notes = note_seqs[sym_ind]
        # print 'notes', notes
        if play:
            self.emit("playNotes", notes)

        if len(notes) > 0:
            text = self.remove_extra_spacing(text)
            text_previous = ''
            if len(self.history) > 1:
                text_previous = self.remove_extra_spacing(self.history[-2])
            self.emit("updateHistory", text_previous, text, )

            self.on_generateAlternatives(text, pos)

    @staticmethod
    def remove_extra_spacing(text):
        parts = text.strip().split(' ')
        non_space_parts = []
        for part in parts:
            non_space_part = part.strip()
            if len(non_space_part) > 0:
                non_space_parts.append(non_space_part)
        return ' '.join(non_space_parts)

    # get the symbol index when given the left side pos of the sym
    def get_sym_ind_from_left(self, text, left):
        count_syms_before = 0
        p_is_space = False
        for i in range(left+1):
            if text[i] == ' ' and i and not p_is_space:
                count_syms_before += 1
                p_is_space = True
            elif text[i] != ' ':
                p_is_space = False
        if not left:
            count_syms_before = 0
        return count_syms_before

    # for alternatives, get symbol that's currently being edited
    def get_sym_at_pos(self, text, pos, return_ind=False):
        # for alternatives, get symbol that's currently being edited
        # symbol that is to the left of the cursor
        if len(text) == 0:
            return None, None
        right = pos-1
        if right < 0:
            right = 0
        right_iter_start = right
        if len(text)-1 <= right_iter_start:
            right_iter_start = len(text)-1
        for i in range(right_iter_start, len(text)):
            if text[i] == ' ':
                right = i
                break
        if pos > 0:
            left = pos-2
            if left < 0:
                left = 0
        else:
            left = 0
        left_iter_start = pos - 2
        if len(text)-1 <= left_iter_start:
            left_iter_start = len(text) - 1
        for i in range(left_iter_start, -1, -1):
            if text[i] == ' ':
                left = i + 1
                break
        # TODO: hacky
        if i == 0:
            left = 0

        sym_ind = self.get_sym_ind_from_left(text, left)

        sym = text[left:right+1]
        sym = sym.strip()
        if len(sym) == 0:
            return None, None
        if self.symbol_type == 'roman':
            sym = self.back_to_roman(sym)
        if return_ind is True:
            return sym, sym_ind
        else:
            return sym

    # for playback, gets symbol before the space
    def get_sym_before_pos(self, text, pos, return_ind=False):
        # minus 1 because the caret pos is to right of a space
        right = pos - 1
        print 'right', right,
        right_char_ind = right
        for i in range(right, -1, -1):
            if text[i] != ' ':
                right_char_ind = i
                break
        right = right_char_ind
        print right
        # find the left boundary
        left = 0
        for i in range(right, -1, -1):
            if text[i] == ' ':
                left = i
                break
        print 'left', left
        sym_ind = self.get_sym_ind_from_left(text, left)
        print 'sym_ind', sym_ind
        sym = text[left:right+1]
        sym = sym.strip()
        # sym = self.back_to_roman(sym)

        if return_ind is True:
            return sym, sym_ind
        else:
            return sym
