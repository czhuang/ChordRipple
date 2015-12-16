

class SuggestionList(object):
    def __init__(self, parent, id):
        self.id = id
        self.items = list()
        # self.types = self.make_suggestion_types()
        self.parent = parent

    @property
    def num_items(self):
        return len(self.items)

    def retrieve_item_at(self, ind, query):
        print 'retrieve_item_at', ind, 'total items', len(self.items)
        if ind >= len(self.items):
            return None
        item = self.items[ind]
        print item.seq
        retrieve_text = ' '.join(item.seq)
        print query.seqStr, len(query.seqStr)
        print retrieve_text, len(retrieve_text)

        # TODO: doesn't check when text is beyond loop len
        if len(query.seq) <= self.parent.loop_len - 1:
            assert query.seqStr.strip() == retrieve_text.strip()
        return item

    def add(self, items):
        assert self.parent is not None
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
        self.kind = kind  # model + type of recommendation
        self.model = self.which_model()
        self.tags = tags

    def which_model(self):
        if self.kind == 'start':
            return 'manual'
        elif self.kind == 'sub_ripple':
            return 'nn+ngram'
        elif self.kind == 'next':
            return 'ngram'
        elif self.kind == 'side_ripple':
            return 'ngram'
        else:
            return None
