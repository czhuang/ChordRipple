
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


class QueryObject(object):
    def __init__(self, query):
        self.seq_str = query["text"]
        self.seq = query["seq"]
        if "activeIdx" not in query:
            self.activeIdx = None
        else:
            self.activeIdx = query["activeIdx"]

        self.author = query["author"]

        if "actionKind" not in query["actionKind"]:
            self.actionKind = None
        else:
            self.actionKind = query["actionKind"]

        self.loop = query["loop"]
        if "play" not in query:
            self.play = False
        else:
            self.play = query["play"]

        if "panelIdx" not in query:
            self.panelIdx = None
        else:
            self.panelIdx = query["panelIdx"]
        if "itemIdx" not in query:
            self.itemIdx = None
        else:
            self.itemIdx = query["itemIdx"]

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




