
import cPickle as pickle

import numpy as np

from socketio.namespace import BaseNamespace
from socketio.mixins import BroadcastMixin

from dynamic_programming_tools import shortest_path
from retrieve_model_tools import retrieve_NGram, retrieve_SkipGramNN
from music21_chord_tools import sym2roman

class Resource(BaseNamespace, BroadcastMixin):
    def initialize(self):
        # Called after socketio has initialized the namespace.
        self.history = []
        self.parsed_seqs = {}
        self.unit_dur = 1.0

        self.ngram = retrieve_NGram()
        self.nn = retrieve_SkipGramNN()

        self.previous_sym = None
        self.previous_sym_ind = None
        self.n_suggestions = 3

        self.suggest_seqs = []
        self.suggest_inds = []

    @property
    def model(self):
        return self.ngram

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

    def parse_seq_as_chords(self, text):
        text = text.strip()
        if text in self.parsed_seqs.keys():
            chords = self.parsed_seqs[text]
            return chords

        parts = text.split(' ')
        chords = []
        for part in parts:
            if len(part) > 0:
                chords.append(self.make_notes(part))
        self.parsed_seqs[text] = chords
        return chords

    def parse_seq_as_syms(self, text):
        text = text.strip()
        parts = text.split(' ')
        parts = [part for part in parts if len(part) > 0]
        return parts

    def on_playSeq(self, text, pos):
        print '---on_parseSeq:', text
        chords = self.parse_seq_as_chords(text)

        durs = [self.unit_dur] * len(chords)
        self.emit('playSeq', chords, durs)

    def get_similar_chords(self, sym, topn):
        similars = self.nn.most_similar(sym, topn=topn)
        if similars is None:
            return
        sims = [s[0] for s in similars]
        return sims

    def on_startSeqs(self):
        print '---on_startSeqs'
        start_syms = []
        limit = 20
        while len(start_syms) < self.n_suggestions:
            sym = self.ngram.sample_start()
            if sym not in start_syms:
                start_syms.append(sym)
        sims = [[sym] for sym in start_syms]
        sim_inds = [[0]] * len(start_syms)
        print sims
        print sim_inds
        self.emit('updateChordSuggestions',
                   sims, sim_inds)

    def clear_suggestions(self):
        self.suggest_seqs = []
        self.suggest_inds = []

    def add_suggestions(self, seqs, inds):
        for i, seq in enumerate(seqs):
            if seq not in self.suggest_seqs:
                print seq, inds[i]
                self.suggest_seqs.append(seq)
                self.suggest_inds.append(inds[i])

    def on_generateAlternatives(self, text, pos):
        print '--- --- generate_alternative --- ---', text, pos
        self.clear_suggestions()
        original_seq = self.parse_seq_as_syms(text)
        sym, sym_ind = self.get_sym_at_pos(text, pos, return_ind=True)
        print sym, sym_ind
        print original_seq
        if self.previous_sym == sym and self.previous_sym_ind == sym_ind:
            return
        else:
            self.previous_sym = sym
            self.previous_sym_ind = sym_ind

        if sym is None or not len(sym):
            print 'WARNING: no symbol at this position'
            return

        if original_seq[sym_ind] != sym:
            print 'WARNING: symbol not matching'
            return

        # generate nexts
        if sym_ind == len(original_seq) - 1:
            ss, sinds = self.generate_next(sym, sym_ind, original_seq)
            for i, s in enumerate(ss):
                print s, sinds[i]
            self.add_suggestions(ss, sinds)

        # generate substitutions
        sims = self.get_similar_chords(sym, 3)
        if sims is None:
            return
        # only replacing one symbol for now
        sims = [[s] for s in sims]
        sub_seqs = [original_seq[:sym_ind] + s + original_seq[sym_ind+1:] for s in sims]
        sim_inds = []
        for s in sims:
            sim_inds.append([sym_ind])
        self.add_suggestions(sub_seqs, sim_inds)

        # generate ripples
        seq_subs, seq_inds = self.generate_more_change_alternatives(text, pos)
        if seq_subs is not None:
            self.add_suggestions(seq_subs, seq_inds)

        if sym_ind == len(original_seq) - 1:
            ss, sinds = self.generate_continuations(sym, sym_ind, original_seq)
            self.add_suggestions(ss, sinds)

        self.emit('updateChordSuggestions',
                   self.suggest_seqs, self.suggest_inds)


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
        syms = self.model.syms
        sym_ind = syms.index(sym)
        n_conts = 3
        inds = np.argsort(-trans[sym_ind, :])[:n_conts]
        subs = [ original_seq + [syms[ind]] for ind in inds]
        return subs, [[seq_ind+1]]*n_conts

    def generate_more_change_alternatives(self, text, pos):
        sym, sym_ind = self.get_sym_at_pos(text, pos, return_ind=True)
        if sym is None:
            return
        original_seq = self.parse_seq_as_syms(text)
        win_max = 2

        sims = self.get_similar_chords(sym, 3)
        sims.insert(0, sym)
        if sims is None:
            return

        seq_subs = []
        seq_inds = []
        for win in range(1, win_max):
            ub = sym_ind+win
            lb = sym_ind-win

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

    def on_textChange(self, text, pos, play=True):
        self.history.append(text)
        if text is None or not isinstance(text, unicode):
            print "WARNING: chord symbol not valid", text
            return
        sym, sym_ind = self.get_sym_before_pos(text, pos, return_ind=True)
        if play:
            midi = self.on_makeNotes(sym)
        else:
            midi = self.make_notes(sym)

        if midi is not None:
            text = self.remove_extra_spacing(text)
            text_previous = ''
            if len(self.history) > 1:
                text_previous = self.remove_extra_spacing(self.history[-2])
            self.emit("updateHistory", text_previous, text)

            self.on_generateAlternatives(text, pos)

    def on_makeNotes(self, sym):
        midi = self.make_notes(sym)
        self.emit("playNotes", midi)
        return midi

    def make_notes(self, sym):
        sym = sym.strip()
        chord_sym = sym2roman(sym)
        midi = []
        if chord_sym is not None:
            midi = [pch.midi for pch in chord_sym.pitches]
            print 'make midi notes:', midi
        return midi

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
        for i in range(left):
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
        for i in range(pos-1, len(text)):
            if text[i] == ' ':
                right = i
                break
        if pos > 0:
            left = pos-2
            if left < 0:
                left = 0
        else:
            left = 0
        for i in range(pos-2, -1, -1):
            if text[i] == ' ':
                left = i + 1
                break
        if i == 0:
            left = 0
        sym = text[left:right+1]
        sym_ind = self.get_sym_ind_from_left(text, left)

        sym = text[left:right+1]
        sym = sym.strip()
        if return_ind is True:
            return sym, sym_ind
        else:
            return sym

    # for playback, gets symbol before the space
    def get_sym_before_pos(self, text, pos, return_ind=False):
        right = pos - 2
        # minus 2 because the caret pos is to right of a space
        # find the left boundary
        right_char_ind = right
        for i in range(right, -1, -1):
            if text[i] != ' ':
                right_char_ind = i
                break
        right = right_char_ind

        left = 0
        for i in range(right, -1, -1):
            if text[i] == ' ':
                left = i
                break

        sym_ind = self.get_sym_ind_from_left(text, left)

        sym = text[left:right+1]
        sym = sym.strip()
        if return_ind is True:
            return sym, sym_ind
        else:
            return sym