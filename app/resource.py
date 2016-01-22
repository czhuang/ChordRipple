
import os

from copy import copy
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

from SuggestionList import SuggestionList, SuggestionItem
from QueryObject import QueryObject

from Logs import Logs


PKL_FOLDER = 'data'

EXPERIMENT_TYPE_STRS = ['Single-T',
                        'Single-A',
                        'Ripple',
                        'Null']
TYPICAL, SIM_BUT_LESS_TYPICAL, RIPPLE, NULL = range(len(EXPERIMENT_TYPE_STRS))
EXPERIMENT_TYPE = RIPPLE

from music21_chord_tools import ROMAN2LETTER_RELABELS
from music21_chord_tools import ROMAN_PARTIAL_RELABELS
from music21_chord_tools import MAKE_NOTE_EXCEPTIONS
from music21_chord_tools import LETTER2ROMAN_RELABELS
from music21_chord_tools import LETTER_PARTIAL_RELABELS_FOR_USER

from latin_squares_experiment_tools import get_condition_ordering

MIDI_START_SLACK = 0.15

from Database import CHORD_LEN
from Database import Database, DatabasePlacebo

ALL_SIMS = True
DISABLE_NEXT = False
DISABLE_SIDE_RIPPLES = True


# TODO: the change in experiment types might cause the logging to break


class Resource(BaseNamespace, BroadcastMixin):
    def initialize(self):
        # Called after socketio has initialized the namespace.
        self.history = []
        self.parsed_seqs_notes = {}
        self.unit_dur = 60/92.0

        self.ngram = retrieve_NGram()
        self.nn = retrieve_SkipGramNN()

        # print self.nn.syms
        assert self.ngram.syms == self.nn.syms

        # self.db = Database('test')
        # self.db = Database('study-iui')
        self.db = DatabasePlacebo()

        self.db.index_model(self.ngram, 'ngram')
        self.db.index_model(self.nn, 'nn')

        # get randomized ordering for current experiment
        condition_orderings = get_condition_ordering()
        fpath = os.path.join('pkls', 'participant_count.txt')
        with open(fpath, 'r') as p:
            print 'reading lines from:', fpath
            participant_count = p.readline()
        try:
            self.participant_count = int(participant_count)
        except:
            print type(participant_count), participant_count

        print 'participant_count:', self.participant_count

        self.participant_count = self.participant_count % condition_orderings.shape[0]
        print 'participant_count modulo:', self.participant_count

        self.ordering = condition_orderings[self.participant_count, :]
        print 'ordering:', self.ordering
        self.emit('ordering', list(self.ordering))

        # self.experiment_type = EXPERIMENT_TYPE
        # for tutorial, always start with ripple
        # self.experiment_type = RIPPLE  # int(self.ordering[0])
        #
        # self.experiment_type = TYPICAL
        self.rec_types = [NULL]
        self.rec_types_history = [self.rec_types]

        query = QueryObject(dict(data=list(self.ordering), actionKind="ordering"))
        self.index_user_action(query)

        query = QueryObject(dict(data=self.participant_count, actionKind="participant_count"))
        self.index_user_action(query)

        # update participant count
        with open(fpath, 'w+') as p:
            p.truncate()
            p.writelines(str(self.participant_count + 1))


        # self.previous_sym = None
        # self.previous_sym_ind = None

        self._previous_sym = None
        self._previous_sym_ind = None

        self.n_suggestions = 5
        self.n_similar = 2

        self.suggestions = SuggestionList(self, 'below')
        self.suggestions_above = SuggestionList(self, 'above')

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

        self.logs = Logs(EXPERIMENT_TYPE, EXPERIMENT_TYPE_STRS)

        self.loop_len = None

        # for first chord
        if self.is_rec_type_active(TYPICAL):
            if self.symbol_type == 'roman':
                self.start_syms = [['I'], ['i'], ['V'], ['IV'], ['I7']]
            else:
                self.start_syms = [['C'], ['Cm'], ['G'], ['F'], ['C7']]
        else:
            # want diverse sequences
            if self.symbol_type == 'roman':
                # self.start_syms = [[u'I', u'vi'], ['V', 'vi'], ['IV', 'ii'], [u'V7/IV', u'IV'], [u'i', u'i'], [u'i', u'V6']]
                self.start_syms = [[u'i'], [u'I'], ['V'], ['iv'], ['IV'], ['V7/IV']]
            else:
                # self.start_syms = [['a', 'C64'], ['G', 'C'], ['A', 'gb'], ['C', 'F'], ['G', 'e'], ['Bb', 'Ab'], ['E', 'G']]
                # self.start_syms = [['C'], ['F'], ['G'], ['Am'], ['Cm'], ['B-'], ['A-']]
                # self.start_syms = [['Cm'], ['Am'], ['B-'], ['F/C'], ['G'], ['Dm7'], ['Cmaj7'], ['F#dim']]
                self.start_syms = [['Cm'], ['Am'], ['Bb'], ['F/C'], ['G'], ['Dm7'], ['Cmaj7'], ['F#dim']]


        # set initial sequence
        # self.init_seqs = [['C', 'F', 'Dm', 'G', 'C', 'C', 'F'],
        #                   ['C', 'F', 'C', 'F', 'G7', 'C', 'G'],
        #                   ['C', 'Am', 'G', 'C', 'F', 'C', 'G'],
        #                     ['C', 'F', 'G', 'C', 'F', 'C', 'F']]

        self.init_seqs = [['C', 'F', 'Dm', 'G', 'C', 'C', 'F', 'C'],
                          ['C', 'F', 'C', 'F', 'G7', 'C', 'G', 'C'],
                          ['C', 'Am', 'G', 'C', 'F', 'C', 'G', 'C'],
                            ['C', 'F', 'G', 'C', 'F', 'C', 'F', 'C']]
        self.on_generate_complete_seq(self.init_seqs[0])

    def is_rec_type_active(self, rec_type):
        if rec_type in self.rec_types:
            return True
        else:
            return False

    def get_active_rec_type_str(self):
        active_rec_type_str = ''
        for rec_type in self.rec_types:
            active_rec_type_str += EXPERIMENT_TYPE_STRS[rec_type] + '_'
        if active_rec_type_str > 1:
            active_rec_type_str[:-1]
        return active_rec_type_str

    def any_rec_type_active(self):
        if TYPICAL not in self.rec_types and SIM_BUT_LESS_TYPICAL not in self.rec_types:
            return False
        return True

    # ======================
    # == database helpers ==
    # ======================
    def index_user_action(self, query, suggestions=None,
                          suggestions_above=None, attrs=None):
        # experiment_type_label = EXPERIMENT_TYPE_STRS[self.experiment_type]
        experiment_type_label = self.get_active_rec_type_str()

        print '...index_user_action:', query.seqStr, query.actionKind, \
            query.author, query.panelId, query.itemIdx

        actionKindThatNeedsItem = 'use' == query.actionKind or 'play' == query.actionKind
        itemInAttrs = attrs is not None and 'item' in attrs
        assert (actionKindThatNeedsItem and itemInAttrs) or not itemInAttrs

        if itemInAttrs:
            print '\t\t\t', attrs['item'].inds, attrs['item'].seq

        suggestions_list = []
        if suggestions is not None:
            suggestions_list.append(suggestions)
        if suggestions_above is not None:
            suggestions_list.append(suggestions_above)
        if len(suggestions_list) == 0:
            suggestions_list = None

        self.db.index_user_action(query, experiment_type_label,
                                  suggestions_list, attrs)


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

    # def on_loopLen(self, loop_len):
    #     self.loop_len = loop_len

    def on_ripple(self, status):
        self.change_experiment_type(RIPPLE, status)

    def add_rec_type(self, rec_type):
        if rec_type not in self.rec_types:
            self.rec_types.append(rec_type)
            self.rec_types_history.append(copy(self.rec_types))

    def remove_rec_type(self, rec_type):
        if rec_type in self.rec_types:
            self.rec_types.remove(rec_type)
            self.rec_types_history.append(copy(self.rec_types))

    def change_experiment_type(self, new_type, status):
        if status:
            self.add_rec_type(new_type)
        else:
            self.remove_rec_type(new_type)
        print 'rec type history', self.rec_types_history
        print 'EXPERIMENT type changed to:', self.get_active_rec_type_str()

    def get_previous_experiment_mode(self):
        # mode gives transition versus sim
        # orthogonal dimension to ripple or not
        for type_ in self.expr_type_history[::-1]:
            if type_ != RIPPLE:
                return type_
        return None

    # def on_rippleOff(self):
    #     previous_mode = self.get_previous_experiment_type()
    #     if previous_mode is not None:
    #         self.change_experiment_type(previous_mode)
    #     else:
    #         self.change_experiment_type(SIM_BUT_LESS_TYPICAL)

    def on_defaultSeq(self):
        self.on_generate_complete_seq(self.init_seqs[0])

    def on_transitionMode(self, status):
        self.change_experiment_type(TYPICAL, status)

    def on_simMode(self, status):
        self.change_experiment_type(SIM_BUT_LESS_TYPICAL, status)

    def on_inputSave(self, text):
        # log add automatically saves upon seeing type "save"
        # self.logs.add("save", text)
        query = QueryObject(dict(text=text, actionKind="save"))
        self.index_user_action(query)

    def on_comments(self, text):
        print '...on_comments', text
        # log add automatically saves upon seeing type "save"
        # self.logs.add("save", text)
        query = QueryObject(dict(text=text, actionAuthor="user",
                                 actionKind="comments"))

        self.index_user_action(query)

    def on_rating(self, lineText, id, value, caption):
        print '...on_rating', lineText, value, id, caption
        # log add automatically saves upon seeing type "save"
        # self.logs.add("save", text)
        query = QueryObject(dict(text=lineText, actionAuthor="user",
                                 actionKind="rating"))
        attrs = {'rating': value,
                 'ratingCaption': caption,
                 'saveIdx': id,
                 'ratingQuestion': lineText}

        query.add_attributes(attrs)

        self.index_user_action(query)

    def on_clear(self, text):
        # self.logs.add("clear", text)
        query = QueryObject(dict(text=text, actionKind="clear"))
        self.index_user_action(query)
        self.clear_suggestions()

    # def on_rating(self, value, ind, text):
    #     try:
    #         ind = int(ind)
    #     except ValueError:
    #         print 'WARNING: index to save entry wanted, but received', ind
    #
    #     if len(text) > 0:
    #         self.logs.add_rating(value, 'save', text, ind)
    #     logs = self.logs.save
    #     for log in logs:
    #         print log

    def on_setPlaybackSpeed(self, speed):
        print '...playbackspeed',
        self.unit_dur = 60.0 / speed
        print self.unit_dur

    def disconnect(self, *a, **kw):
        super(Resource, self).disconnect(*a, **kw)

    def on_ping(self, param, loop_len):
        print '---on_ping', param, loop_len
        self.loop_len = loop_len
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

    # def parse_seq_as_notes(self, text):
    #     text = text.strip()
    #     if text in self.parsed_seqs_notes.keys():
    #         chords = self.parsed_seqs_notes[text]
    #         return chords
    #     parts, raw_parts = self.parse_seq_as_syms(text)
    #     chords = self.make_note_seqs(parts)
    #     self.parsed_seqs_notes[text] = chords
    #     return chords
    #


    def seqToNotes(self, seq, text):
        # chords are list of list of notes
        chords = self.make_note_seqs(seq)
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

    # def make_log_tags_for_playback(self, query):
    #     tags = {}
    #     print 'make_log_tags_for_playback', query.author
    #     if query.author == 'machine':
    #         item = self.retrieve_suggestion_item_at(query)
    #         if item is None:
    #             print 'WARNNG: can not retrieve suggestion item', query.activeIdx, \
    #                 query.seqStr
    #             assert False
    #         tags['suggestionItem'] = item
    #     # tags['author'] = author
    #     return tags

    # def on_playSeq(self, text, pos, author, activeIdx,
    #                suggestPanel=None, loop=False):

    def preprocess_query(self, query):

        return query

    def on_playSeq(self, original_query):
        print "on_playSeq:" #, original_query
        query = QueryObject(original_query)

        # print '...on_playSeq, loop', query.loop

        # logging
        # TODO: re-enable logging later, has some problems
        # need to make sure the chords played are actually the chords written
        if query.author == 'dont_log':
            pass
        else:
            if query.loop:
                assert query.actionKind == 'loop'
            else:
                assert query.actionKind == 'play' or query.actionKind == 'play both'

            attrs = self.retrieve_suggestion_item_as_attrs(query)
            self.index_user_action(query, self.suggestions,
                                   self.suggestions_above, attrs)
            # self.logs.add(query.actionKind, query.seqStr, tags)

        midi_notes = self.make_proxy_midi_notes_from_query(query)

        if query.actionKind == "artificalPlay":
            self.emit('playSeq', midi_notes, False, original_query)
        else:
            self.emit('playSeq', midi_notes, True, original_query)#, query.play)

    def make_proxy_midi_notes_from_query(self, query):
        print '---on_parseSeq:', query.seqStr
        chords = self.seqToNotes(query.seq, query.seqStr)
        print 'durations', query.durations

        if query.durations is not None:
            durs = [1.0*dur for dur in query.durations]
            print 'len(chords)', len(chords)
            print 'scaled durs', len(durs), durs
            #assert len(durs) == len(chords)
        else:
            durs = [self.unit_dur] * len(chords)

        print '...cumulative durations'
        start_time_acc = [0]
        for dur in durs:
            start_time_acc.append(start_time_acc[-1]+dur)
        print start_time_acc

        if query.loop:
            chords = chords[:] + chords[:]
            durs = durs[:-1] + [durs[-1]+self.unit_dur*0.2] + durs[:]

        midi_notes = self.make_proxy_midi_notes(chords, durs)
        return midi_notes


    def make_proxy_midi_notes(self, chords, durs):
        if not len(chords):
            return []
        if not isinstance(chords[0], list):
            chords = [chords]
        midi_notes = []
        running_time = 0.0
        for i, notes in enumerate(chords):
            onset = running_time
            if i == 0:
                onset += MIDI_START_SLACK
            if i >= len(durs):
                dur = durs[-1]
            else:
                dur = durs[i]
            for note in notes:
                midi_note = {'pitch': note, 'onset': onset,
                             'offset': running_time+dur}
                midi_notes.append(midi_note)
            running_time += dur
        return midi_notes

    # def on_(self, chord_changes, author, activeIdx,
    #                   suggestPanel, original_query, log=True):
    def on_playSubseq(self, original_query, log=True, playContext=False):
        print "...on_playSubseq, playContext", playContext  #, original_query
        # print '...on_playSubseq', chord_changes, author, activeIdx, log, suggestPanel
        # print chord_changes
        syms = []
        inds = []
        # TODO: assumes continuous but side ripples are not
        durs = []
        # if not isinstance(original_query, QueryObject):
        query = QueryObject(original_query)
        query.playContext = playContext


        query_durations = query.durations
        chord_changes = query.chordSeqsAndFormat
        print chord_changes

        if not playContext:
            for i, chord_change in enumerate(chord_changes[:CHORD_LEN]):
                if chord_change[1]:
                    syms.append(chord_change[0])
                    inds.append(i)
                    if query_durations is not None:
                        durs.append(query_durations[i])
        else:
            start_ind = None
            for i, chord_change in enumerate(chord_changes[:CHORD_LEN]):
                if chord_change[1]:
                    start_ind = i
                    break
            end_ind = None
            for i in range(CHORD_LEN-1, -1, -1):
                if chord_changes[i][1]:
                    end_ind = i
                    break
            print "start_ind, end_ind", start_ind, end_ind
            print "boundary"
            if start_ind - 1 >= 0:
                start_ind -= 1
            if end_ind + 1 < CHORD_LEN:
                end_ind += 1
            print start_ind, end_ind
            
            syms = [ chord_changes[i][0] for i in range(start_ind, end_ind+1) ]
            durs = [ query_durations[i] for i in range(start_ind, end_ind+1) ]
            inds = range(start_ind, end_ind+1)
            print syms, durs
        

        print len(inds), inds
        if len(inds) > 1:
            p_ind = inds[0]
            fixed_inds = inds[:]
            for ind in inds[1:]:
                if ind - p_ind != 1:
                    print 'WARNING: Changes not continuous'
                    for i in range(p_ind+1, ind):
                        fixed_inds.append(i)
                p_ind = ind
            inds = fixed_inds
            # not necessary to sort
            inds.sort()

            # update durations, syms
            # to fulfill the later check of
            # len(durs) != len(syms)

            durs = [query_durations[i] for i in inds]
            syms = [chord_changes[i][0] for i in inds]

        if query.log and log:
            attrs = self.retrieve_suggestion_item_as_attrs(query)
            if query.author == 'machine':
                self.index_user_action(query, self.suggestions,
                                       self.suggestions_above, attrs)
            else:
                self.index_user_action(query, )
            # self.logs.add("play bold", text, tags)

        # original subseq notes
        # need context to determine octave
        all_notes = self.make_note_seqs(query.seq)


        print 'on_playSubseq', len(all_notes), query.seqStr, inds
        notes = [all_notes[ind] for ind in inds]

        # TODO: this should not happen, should check for this, instead of letting it slide
        if len(durs) != len(syms):
            durs = [self.unit_dur] * len(syms)

        midi_notes = self.make_proxy_midi_notes(notes, durs)

        # for entire sequence
        # midi_notes = self.make_proxy_midi_notes_from_query(query)

        # sending the original query back so that have context
        # TODO: make all communication to be query objects
        # with more fields
        self.emit('playSubseq', midi_notes, original_query)
        return notes


    def get_similar_chords(self, sym, topn):
        similars = self.nn.most_similar(sym, topn=topn)
        print self.nn.syms
        if similars is None:
            return
        sims = [s[0] for s in similars]
        return sims

    def on_startSeqs(self):
        print '---on_startSeqs'
        # TODO: fixed start seqs

        self.clear_suggestions()
        for s in self.start_syms:
            self.suggestions.add(SuggestionItem(s, range(len(s)), 'start'))
        seqs, inds = self.suggestions.get_seqs_inds()
        # print seqs
        # print inds
        print '----- updateChordSuggestions ----, # of items', self.suggestions.num_items
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

        # sorted_probs, sorted_syms = \
        #     simple_foward_backward_gap_dist(self.model, before_sym, after_sym,
        #                                     self.experiment_type)

        if self.is_rec_type_active(TYPICAL):
            relevant_rec_type = TYPICAL
        else:
            relevant_rec_type = None

        sorted_probs, sorted_syms = \
                simple_foward_backward_gap_dist(self.model, before_sym, after_sym, relevant_rec_type)


        if sorted_syms is not None and len(sorted_syms) > 10:
            for i in range(10):
                print sorted_syms[i], np.exp(sorted_probs[i])
            ind = sorted_syms.index('C')
            print "what is prob for C?", np.exp(sorted_probs[ind]), ind

        # TODO: this is a hack for the beginning when the dynamic programming doesn't take "starting prior" into consideration
        if sorted_syms is None and sym_ind == 0 and self.is_rec_type_active(TYPICAL):
            sorted_syms = [s[0] for s in self.start_syms]

        n_subs = factor*self.n_similar
        if sorted_syms is not None:
            subs = sorted_syms[:n_subs]
        if original_sym in subs:
            subs.remove(original_sym)
            subs.append(sorted_syms[n_subs])
        print '...subs', subs
        return sorted_syms, subs

    def make_single_sub_suggestion_items(self, sym_ind, original_seq,
                                         subs, sorted_syms, return_tags_only=False):
        # original_sym = original_seq[sym_ind]
        suggestion_items = []
        tags_list = []
        for i, ss in enumerate(subs):
            sub_seq = original_seq[:sym_ind] + [ss] + original_seq[sym_ind + 1:]
            # print 'subseq', sub_seq
            tags = {}
            if i < self.n_similar:
                tags['source'] = 'subs'
            else:
                tags['source'] = 'sim'

            # if sorted_syms is not None:
            # TODO: hack for now, fix sorted_syms later
            if sorted_syms is not None and ss in sorted_syms:
                # print len(sorted_syms), ss
                # print sorted_syms
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
        if not self.any_rec_type_active():
            return None, None

        original_sym = original_seq[sym_ind]
        print '...generate substitutions based on similarity'
        # generate substitutions based on similarity
        # if self.experiment_type != TYPICAL:  # and sym_ind == 0:

        if self.is_rec_type_active(SIM_BUT_LESS_TYPICAL):
            sims = self.get_similar_chords(original_sym, self.n_similar*2)
        # elif self.experiment_type != TYPICAL:
        #     sims = self.get_similar_chords(original_sym, self.n_similar)
        else:
            sims = None
        print "sims", sims
        # generate substitutions based on context
        if self.is_rec_type_active(TYPICAL):
            factor = 2

        # sorted_syms is for in case needed more?
        # if not typical, then for first chord, only use sim

        # if self.experiment_type != TYPICAL:  # and sym_ind == 0:
        # TODO: if both typical and sim are active, sim doesn't use sorted_syms, may cause problems for later functions that use this
        sorted_syms = None
        subs = None

        if self.is_rec_type_active(TYPICAL):
            sorted_syms, subs = self.generate_subs_from_context(sym_ind, original_seq,
                                                            factor=factor)
        print "subs by context", subs
        # collect all the single changes
        if subs is None:
            subs = sims
        elif sims is not None:
            subs.extend(sims)
        # subs first, sims next
        print "all collected singletons", subs
        if self.is_rec_type_active(TYPICAL) and self.is_rec_type_active(SIM_BUT_LESS_TYPICAL):
            assert len(subs) == 8 or subs is None or len(subs) == 0
        elif self.is_rec_type_active(TYPICAL) or self.is_rec_type_active(SIM_BUT_LESS_TYPICAL):
            assert len(subs) == 4 or subs is None or len(subs) == 0

        # sorted_syms is for in case needed more?
        # not currently used in the calling function

        return sorted_syms, subs

    @property
    def previous_sym(self):
        return self._previous_sym

    @previous_sym.setter
    def previous_sym(self, sym):
        print '...setting previous sym to:', sym
        self._previous_sym = sym

    @property
    def previous_sym_ind(self):
        return self._previous_sym_ind

    @previous_sym_ind.setter
    def previous_sym_ind(self, ind):
        print '...setting previous ind to:', ind
        self._previous_sym_ind = ind

    def on_generateAlternatives(self, query, log=True):

        # if experiment type is MANUAL then don't give any alternatives
        # if self.experiment_type == TYPICAL:
        #     return
        print '\n\n--- --- generate_alternative --- ---',
        # print self.symbol_type
        # print 'on_generateAlternatives, query', query
        if not isinstance(query, QueryObject):
            query = QueryObject(query)
        # if sequence is empty, empty recommendations, add start-seq recommendations, and return
        if len(query.seq) == 0:
            self.previous_sym = None
            self.previous_sym_ind = None
            self.clear_suggestions()
            self.emit('updateChordSuggestions', [], [], [])
            self.on_startSeqs()
            return

        # check previously active chord, could be None
        print 'previous', self.previous_sym, self.previous_sym_ind
        print 'query', query.sym
        # if previuos symbol is the same as currently actively symbol
        # then don't need to generate new alternatives
        if self.previous_sym is not None and self.previous_sym_ind == query.activeIdx \
                and self.previous_sym == query.sym:
            return

        # if current symbol is empty and next symbol is also empty
        # don't do anything
        if len(query.sym) == 0 and query.activeIdx is not None \
                and len(query.seq[query.activeIdx]) == 0:
            return

        # index here?
        self.index_user_action(query, self.suggestions,
                               self.suggestions_above)

        self.clear_suggestions()
        self.previous_sym = query.sym
        self.previous_sym_ind = query.activeIdx

        original_sym = query.sym
        raw_original_seq = query.seq
        original_seq = query.seq
        sym_ind = query.activeIdx
        # generate new alternatives
        if original_sym is None or not len(original_sym):
            print 'WARNING: no symbol at this position'
            # if sym in middle, then use context to ripple in to generate suggestions
            sorted_syms, subs = self.generate_subs_from_context(sym_ind,
                                                                raw_original_seq, 4)
            if sorted_syms is not None:
                suggestion_items = self.make_single_sub_suggestion_items(sym_ind, raw_original_seq,
                                                                         subs, sorted_syms)
                self.clear_suggestions()
                self.suggestions_above.add(suggestion_items)
                seqs, inds = self.suggestions_above.get_seqs_inds()

                print '...generateAlternatives, # of items', self.suggestions_above.num_items
                self.emit('updateChordSuggestionsAbove', seqs, inds)#, self.suggestions_above.types)
            return

        # bottom, next ripples
        print '...generate_nexts', raw_original_seq
        print original_seq, original_sym

        if not DISABLE_NEXT:
            lastPos = query.activeIdx + 1 == len(original_seq)

            nextSymIsEmpty = not lastPos and \
                             len(original_seq[query.activeIdx+1]) == 0
            # if sym_ind == len(original_seq) - 1 or nextSymIsEmpty:
            # print 'sym_ind', sym_ind, CHORD_LEN, nextSymIsEmpty

            if sym_ind < CHORD_LEN - 1 or nextSymIsEmpty:
                ss, sinds = self.generate_next(original_sym, sym_ind, raw_original_seq)
                if ss is None:
                    print 'WARNING: no next chords for ', original_seq[sym_ind]
                    return
                for i, s in enumerate(ss):
                    self.suggestions.add(SuggestionItem(s, sinds[i], 'next'))
                    print s, sinds[i]

        # bottom, side ripples
        if self.is_rec_type_active(RIPPLE) and not DISABLE_SIDE_RIPPLES:
            suggestion_items = self.generate_side_ripples(sym_ind, original_seq)
            self.suggestions.add(suggestion_items)

        # above, single sims and subs by context
        sorted_syms, subs = self.generate_singleton_subs(sym_ind, raw_original_seq)
        print '...generate_alternatives, sorted_syms', sorted_syms

        if subs is None:
            seqs, inds = self.suggestions_above.get_seqs_inds()
            self.emit('updateChordSuggestionsAbove', seqs, inds)
            return
        else:
            suggestion_items = self.make_single_sub_suggestion_items(sym_ind, raw_original_seq, subs, sorted_syms)

        # generate ripples for the single changes
        print '...subs', subs
        if self.is_rec_type_active(RIPPLE):
            seq_subs, seq_inds = self.generate_ripples(raw_original_seq, sym_ind,
                                                       subs, all_sims=ALL_SIMS)
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
            print "len(suggestion_items) == len(seq_subs)?"
            print len(suggestion_items), len(seq_subs)
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
        self.emit('updateChordSuggestions', seqs, inds)#, self.suggestions.types)

        seqs, inds = self.suggestions_above.get_seqs_inds()
        print '...generateAlternatives, # of above items', self.suggestions_above.num_items
        print seqs
        print inds
        self.emit('updateChordSuggestionsAbove', seqs, inds)#, self.suggestions.types)

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

    def on_generate_complete_seq(self, seq=None):
        if seq is None:
            seq = self.ngram.gen_seq(CHORD_LEN)
        seq, seq_original = self.format_seq(seq)
        # seq = [u'C', u'F', u'D', u'G', u'C', u'C', u'F']

        seq_str = ' '.join(seq)
        print 'seq_str', seq_str

        query = QueryObject(dict(text=seq_str, author="machine",
                                 actionAuthor="machine",
                                 actionKind="start_complete_seq"))
        self.index_user_action(query)
        self.emit('set_seq', seq_str)

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

        # subs = [original_seq[:] + [syms[ind]] for ind in inds]
        subs = [original_seq[:seq_ind+1] + [syms[ind]] + original_seq[seq_ind+2:] for ind in inds]
        # print 'generate_next', subs
        return subs, [[seq_ind+1]]*n_conts

    def generate_ripples(self, original_seq, sym_ind, sims, win_max=2,
                         all_sims=False):
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

            ub += 1
            lb -= 1

            if lb < 0:
                lb = 0
            if ub > len(original_seq):
                ub = len(original_seq)

            for j, s in enumerate(sims):
                fixed = {}

                for idx in range(lb, ub+1):
                    if idx < len(original_seq):
                        fixed[idx] = original_seq[idx]

                # may override lb or ub
                fixed[sym_ind] = s
                print fixed

                # hack for if sequence comes in with empty trailing
                # spaces that causes one to think that the sequence is longer
                max_ind = np.max(fixed.keys())
                # if last two index is empty than take out the max
                if len(fixed[max_ind]) == 0 and max_ind-1 in fixed.keys() \
                    and len(fixed[max_ind-1]) == 0:
                    del fixed[max_ind]

                if not all_sims:
                    sub_seq, sym_inds = \
                        shortest_path(self.model, fixed, sym_ind, original_seq)
                else:
                    sub_seq, sym_inds = \
                        shortest_path(self.model, fixed, sym_ind,
                                      original_seq, self.nn)

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

    # def generate_more_change_alternatives(self, text, pos):
    #     sym, sym_ind = self.get_sym_at_pos(text, pos, return_ind=True)
    #     if sym is None:
    #         return
    #     original_seq, raw_original_seq = self.parse_seq_as_syms(text)
    #     win_max = 2
    #
    #     sims = self.get_similar_chords(sym, 3)
    #     sims.insert(0, sym)
    #     if sims is None:
    #         return
    #     return self.generate_ripples(raw_original_seq, sym_ind, sims, win_max)

    def on_next(self, experiment_count):
        print '--- on_next', experiment_count
        # experiment_count starts at 1
        self.clear_suggestions()

        # emit rating questions
        self.questions = {0: [['1. The tool helped me explore a wider range of chords.']],
                            1: [['2. Ripples made it easier to adopt difficult chords.']],
                            2: [["3. I'm happy with the chord progressions I came up with."]]}

        self.questions = {0: [], 1:[], 2:[]}
        # "How did you use ripples?"
        experiment_idx = experiment_count - 1
        if experiment_idx < len(self.ordering):
            which_experiment = int(self.ordering[experiment_idx])
            print "self.ordering", self.ordering
            print "experiment_idx", experiment_idx
            print "which experiment ind", which_experiment
            print 'which_experiment', EXPERIMENT_TYPE_STRS[which_experiment]
            self.experiment_type = which_experiment
            self.emit('survey', self.questions[which_experiment])

            # use the raw experiment count, since one seq for tutorial
            self.on_generate_complete_seq(self.init_seqs[experiment_count])
        else:
            end_msg = ["This is the end of the experiment.  Thank you for participating!  We really appreciate it and we hope you had some fun too!"]
            self.emit('survey', end_msg)


    def clear_suggestions(self):
        print '... clear_suggestions ...'
        self.previous_sym = None
        self.previous_sym_ind = None
        self.suggestions.clear()
        self.suggestions_above.clear()
        self.emit('updateChordSuggestions', [], [], [])
        self.emit('updateChordSuggestionsAbove', [], [], [])
        return

    def retrieve_suggestion_item_as_attrs(self, query):
        attrs = {}
        if query.author == 'machine':
            ind = query.itemIdx
            print 'suggestions', query.panelId, query.itemIdx, \
                self.suggestions_above.num_items, self.suggestions.num_items

            if 'above' in query.panelId:
                item = self.suggestions_above.retrieve_item_at(ind, query)
            else:
                item = self.suggestions.retrieve_item_at(ind, query)

            # for debugging
            if item is None:
                print 'Error: can not retrieve suggestion item', \
                    query.activeIdx, query.seqStr
                assert False

            attrs = dict(suggestionItem=item)
        return attrs

    def getLastNonEmptyIdx(self, seq):
        pass


    # def on_textChange(self, text, pos, kind, author, ind, suggestPanel=None, play=True):
    def on_textChange(self, original_query):
        print '--- on_textChange ---' #, original_query
        print 'suggestion lists lengths:', self.suggestions_above.num_items, \
            self.suggestions.num_items

        query = QueryObject(original_query)

        if len(query.seqStr) == 0:
            # TODO: why need to emit playSubseq, to set somethings to empty?
            self.emit("playSubseq", [])
            self.clear_suggestions()
            return

        # can't clear suggestions here because need to check use against the suggestion shortly after

        # can have two kinds of use
        # - use the machine-authored chord recommendation
        # - use the user's previous chord sequences

        attrs = None
        if query.actionKind == 'use' and query.author == 'machine':
            assert query.panelId is not None
            attrs = self.retrieve_suggestion_item_as_attrs(query)
        self.index_user_action(query, self.suggestions,
                               self.suggestions_above, attrs)

        # self.logs.add(query.actionKind, query.seqStr, tags)

        # simple history, which actually may be quite sufficient,
        # just don't know when is athor edit and when is machine suggestion used
        self.history.append(query.seqStr)

        # if query.seqStr is None or not isinstance(query.seqStr, unicode):
        if query.seqStr is None:
            print "WARNING: chord symbol not valid", query.seqStr
            return

        if query.sym is None:
            print "WARNING: no chord symbol at position"
            return

        if query.activeIdx is None:
            if query.actionKind == 'use':
                # don't log this extra system playback event
                log = False
                notes = self.on_playSubseq(original_query, log)
                query.play = False
            else:
                return
        else:
            # play activeIdx
            print 'query.seq', query.seq
            chords = self.make_note_seqs(query.seq)
            print 'chords', chords, query.activeIdx
            notes = chords[query.activeIdx]
            print 'notes', notes

            # for entire sequence
            # midi_notes = self.make_proxy_midi_notes_from_query(query)

            # TODO: would there be a case where activeIdx is not None and we do not want to play
            if query.play:
                # if there's a textChange and is by machine,
                # then it's the automatic playback when user chooses to use a suggestion
                if query.author == 'machine':
                    assert query.chordSeqsAndFormat != None
                    # don't log this extra system playback event
                    log = False
                    self.on_playSubseq(original_query, log)
                else:
                    # this is the playback single chord case
                    # i.e. for example if typing in chords

                    if query.durations is not None:
                        durs = [query.durations[query.activeIdx]]
                    else:
                        durs = [self.unit_dur]
                    print 'durs', durs
                    midi_notes = self.make_proxy_midi_notes(notes, durs)
                    print 'midi_notes', midi_notes

                    self.emit("playSubseq", midi_notes, original_query)

        self.clear_suggestions()

        if len(notes) > 0:
            text_previous = ''
            if len(self.history) > 1:
                text_previous = self.remove_extra_spacing(self.history[-2])
            self.emit("updateHistory", text_previous, query.seqStr)


            self.on_generateAlternatives(query)

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
