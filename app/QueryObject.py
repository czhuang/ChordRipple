
from copy import copy


class QueryObjectBk(object):
    def __init__(self, seq, ind, sym=None):
        if sym is not None:
            assert seq[ind] == sym
        self.seq = seq
        self.ind = ind

    @property
    def sym(self):
        return self.seq[self.ind]

    def sub(self, sub, replace=False):
        ind = self.ind
        seq = self.seq
        if isinstance(sub, list):
            seq_sub = seq[:ind] + sub + seq[ind+1:]
        else:
            seq_sub = seq[:ind] + [sub] + seq[ind+1:]
        if replace:
            self.seq = seq_sub
        return seq_sub

# QUERY_ATTR = [("seqStr", str), ("activeIdx", int),
#               ("durations", list),
#               ("actionKind", str),
#               ("loop", int), ("play", int),
#               ("panelIdx", int), ("itemIdx", int),
#     ("sym", str)]


class QueryObject(object):
    def __init__(self, query):
        if "data" not in query:
            self.data = None
        else:
            self.data = query["data"]

        if "text" not in query:
            self.seqStr = None
        else:
            self.seqStr = query["text"].strip()

        if "seq" not in query and self.seqStr is not None:
            self.seq = [sym.strip() for sym in self.seqStr.strip().split(' ')
                        if len(sym.strip()) > 0]
        elif "seq" in query:
            self.seq = query["seq"]
        else:
            self.seq = None

        if "log" not in query:
            self.log = True
        else:
            self.log = query["log"]

        if "chordSeqsAndFormat" not in query:
            self.chordSeqsAndFormat = None
        else:
            self.chordSeqsAndFormat = query["chordSeqsAndFormat"]

        if "activeIdx" not in query:
            # assume if no activeIdx then it's the last
            # self.activeIdx = len(self.seq) - 1
            self.activeIdx = None
        else:
            self.activeIdx = query["activeIdx"]

        self.sym = None
        if self.activeIdx is not None and self.activeIdx < len(self.seq):
            print 'activeIdx', self.activeIdx, 'seq len:', len(self.seq)
            # TODO: hacking a fix for input box focus advancement
            if self.activeIdx >= len(self.seq):
                if self.activeIdx > 0:
                    self.activeIdx -= 1
                else:
                    self.activeIdx = 0
                if len(self.seq) > 0:
                    print 'self.activeIdx', self.activeIdx
                    self.sym = self.seq[self.activeIdx]
            else:
                self.sym = self.seq[self.activeIdx]

        if "durations" not in query:
            self.durations = None
        else:
            self.durations = query["durations"]

        print 'self.activeIdx', self.activeIdx
        if self.seq is not None:
            print 'self.seq', len(self.seq), self.seq

        if "author" not in query:
            self.author = None
        else:
            self.author = query["author"]

        if "actionKind" not in query:
            self.actionKind = None
        else:
            self.actionKind = query["actionKind"]

        if "actionAuthor" not in query:
            self.actionAuthor = None
        else:
            self.actionAuthor = query["actionAuthor"]

        if "loop" not in query:
            self.loop = False
        else:
            self.loop = query["loop"]

        if "play" not in query:
            self.play = False
        else:
            self.play = query["play"]

        if "panelId" not in query:
            self.panelId = None
        else:
            self.panelId = query["panelId"]

        if "itemIdx" not in query:
            self.itemIdx = None
        else:
            self.itemIdx = query["itemIdx"]

        if "originalText" not in query:
            self.originalText = None
        else:
            self.originalText = query["originalText"]

    # @property
    # def sym(self):
    #     return self.seq[self.activeIdx]

    def sub(self, sub, replace=False):
        ind = self.activeIdx
        seq = self.seq
        if isinstance(sub, list):
            seq_sub = seq[:ind] + sub + seq[ind+1:]
        else:
            seq_sub = seq[:ind] + [sub] + seq[ind+1:]
        if replace:
            self.seq = seq_sub
        return seq_sub

    def add_attributes(self, attrs):
        for key, val in attrs.iteritems():
            setattr(self, key, val)
            





