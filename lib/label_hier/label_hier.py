import os
from math import log
import numpy as np


class LabelNode(object):
    def __init__(self, name, index, is_raw):
        self._weight = 0
        self._index = index
        self._name = name
        self._hypers = set()
        self._children = set()
        self._is_raw = is_raw
        self._info_ratio = -1
        self._depth_ratio = -1
        self._freq = -1

    def __str__(self):
        return self._name

    def depth_ratio(self):

        def top2curr(node):
            if len(node.hypers()) == 0:
                return 0
            else:
                return 1 + top2curr(node.hypers()[0])

        def curr2bottom(node):
            if len(node.children()) == 0:
                return 0

            max_d = 0
            for child in node.children():
                max_d = max((max_d, curr2bottom(child)))
            return max_d + 1

        if self._depth_ratio != -1:
            return self._depth_ratio
        else:
            t2c = top2curr(self)
            c2d = curr2bottom(self)
            return 1.0 * t2c / max(t2c + c2d, 0.01)



    def info_ratio(self, leaf_sum):

        def leaf_num(start):
            if len(start.children()) == 0:
                return 1
            else:
                n = 0
                for c in start.children():
                    n += leaf_num(c)
                return n

        if self._info_ratio != -1:
            return self._info_ratio
        else:
            n_leaf_child = leaf_num(self)
            info_ratio = log(leaf_sum, leaf_sum) - log(n_leaf_child, leaf_sum)
            self._info_ratio = info_ratio ** 0.5
        return info_ratio

    def set_index(self, i):
        self._index = i

    def score(self, pred_ind):
        if not self._is_raw:
            print('[sunx] ATTENTION: class index error !!!')
            return -1
        best_score = 0
        gt_paths = self.hyper_paths()
        for h_path in gt_paths:
            for i, h_node in enumerate(h_path):
                if h_node.index() == pred_ind:
                    best_score = max((i+1.0) / (len(h_path)), best_score)
                    break
        return best_score

    def depth(self):
        h_paths = self.hyper_paths()
        dep = 0
        for path in h_paths:
            dep = max(dep, len(path))
        return dep

    def set_raw(self, is_raw):
        self._is_raw = is_raw

    def is_raw(self):
        return self._is_raw

    def name(self):
        return self._name

    def hypers(self):
        return list(self._hypers)

    def add_hyper(self, hyper):
        self._hypers.add(hyper)

    def del_hyper(self, hyper):
        if hyper in self._hypers:
            self._hypers.remove(hyper)

    def children(self):
        return list(self._children)

    def add_child(self, child):
        if child not in self._children:
            self._children.add(child)

    def del_child(self, child):
        if child in self._children:
            self._children.remove(child)

    def index(self):
        return self._index

    def trans_hyper_inds(self):
        hyper_inds = []
        unique_hyper_inds = set()
        h_paths = self.hyper_paths()
        for p in h_paths:
            for w in p:
                if w.index() not in unique_hyper_inds:
                    unique_hyper_inds.add(w.index())
                    hyper_inds.append(w.index())
        return hyper_inds

    def hyper_paths(self):
        if len(self._hypers) == 0:
            # root
            return [[self]]
        else:
            paths = []
            for hyper in self._hypers:
                sub_paths = hyper.hyper_paths()
                for sub_path in sub_paths:
                    sub_path.append(self)
                    paths.append(sub_path)
            return paths

    def show_hyper_paths(self):
        paths = self.hyper_paths()
        for p in paths:
            str = []
            for n in p:
                str.append(n._name)
                str.append('->')
            str = str[:-1]
            print(' '.join(str))

    def weight(self):
        return self._weight

    def set_weight(self, w):
        self._weight = w

    def set_freq(self, freq):
        self._freq = freq

    def freq(self):
        return self._freq


class LabelHier:

    def background(self):
        return self.get_node_by_index(0)

    def root(self):
        raw_label_node = self._label2node[self._raw_labels[-1]]
        return raw_label_node.hyper_paths()[0][0]

    def label2index(self):
        l2i = dict()
        for l in self._label2node:
            l2i[l] = self._label2node[l].index()
        return l2i

    def index2label(self):
        i2l = []
        for n in self._index2node:
            i2l.append(n.name())
        return i2l

    def get_raw_indexes(self):
        raw_inds = []
        for raw_label in self._raw_labels:
            raw_node = self.get_node_by_name(raw_label)
            raw_inds.append(raw_node.index())
        return raw_inds

    def raw2path(self):
        if self._raw2path is None:
            r2p = {}
            for r in self.get_raw_indexes():
                rn = self.get_node_by_index(r)
                r2p[r] = rn.trans_hyper_inds()
            self._raw2path = r2p
        return self._raw2path

    def label_sum(self):
        return len(self._index2node)

    def pos_leaf_sum(self):
        return len(self._raw_labels) - 1

    def get_all_labels(self):
        all_labels = [node.name() for node in self._index2node]
        return all_labels

    def get_raw_labels(self):
        return self._raw_labels

    def get_node_by_name(self, name):
        if name in self._label2node:
            return self._label2node[name]
        else:
            return None

    def get_node_by_index(self, index):
        if index < len(self._index2node):
            return self._index2node[index]
        else:
            return None

    def neg_num(self):
        return self.label_sum() - self.max_depth

    def raw_neg_mask(self, raw_inds):
        # ATTENTION: +Inf value can cause NaN error
        mask = np.ones((raw_inds.shape[0], self.label_sum()))
        for i, raw_ind in enumerate(raw_inds):
            hyper_inds = self.get_node_by_index(raw_ind).trans_hyper_inds()[:-1]
            mask[i, hyper_inds] = float('+Inf')
        return mask

    def _prepare_pos_negs(self):
        raw2pns = {}
        all_inds = set(range(self.label_sum()))
        for raw_ind in self.get_raw_indexes():
            raw_node = self.get_node_by_index(raw_ind)
            all_pos_inds = set(raw_node.trans_hyper_inds())
            all_neg_inds = list(all_inds - all_pos_inds)
            raw2pns[raw_ind] = [raw_ind] + all_neg_inds[:self.neg_num()]

    def _load_raw_label(self, raw_label_path):
        labels = []
        if os.path.exists(raw_label_path):
            with open(raw_label_path, 'r') as f:
                raw_lines = f.readlines()
                for line in raw_lines:
                    labels.append(line.strip('\n'))
        else:
            print('Raw label file not exists !')
        return labels

    def _compress(self):
        self._dfs_compress(self.root())
        # reindex nodes
        new_index = 0
        new_index2node = []
        new_label2node = dict()
        for i, node in enumerate(self._index2node):
            if node is not None:
                node.set_index(new_index)
                new_index2node.append(node)
                new_label2node[node.name()] = node
                new_index += 1

        self._index2node = new_index2node
        self._label2node = new_label2node

        # remove dead children
        # check one parent
        all_labels = set(self.get_all_labels())
        for node in self._index2node:
            hypers = node.hypers()
            assert len(hypers) < 2
            for c in node.children():
                if c.name() not in all_labels:
                    node.del_child(c)

    def _dfs_compress(self, curr):
        # dfs based compression
        if len(curr.children()) > 1:
            # keep curr
            for child in curr.children():
                self._dfs_compress(child)
        elif len(curr.children()) == 1:
            if not curr.is_raw():
                # remove curr
                hypers = curr.hypers()
                for h in hypers:
                    # curr.hyper -> curr.children
                    h.del_child(curr)
                    h.add_child(curr.children()[0])
                    # curr.children -> curr.hyper
                    curr.children()[0].add_hyper(h)

                curr.children()[0].del_hyper(curr)
                self._index2node[curr.index()] = None
                self._label2node[curr.name()] = None

            self._dfs_compress(curr.children()[0])
        else:
            return

    def _check_dead_node(self):
        root = self.root()
        background = self.get_node_by_name('__background__')
        c = 1
        for i in range(self.label_sum()):
            n = self.get_node_by_index(i)
            if len(n.hypers()) == 0 and n.index() != root.index() and n.index() != background.index():
                print('%d: <%s> is dead. (no parent)' % (c, n.name()))
                c += 1
            if len(n.children()) == 0 and not n.is_raw():
                print('%d: <%s> is dead. (no children)' % (c, n.name()))
                c += 1
            if len(n.children()) == 1 and not n.is_raw():
                print('%d: <%s> is dead. (one children)' % (c, n.name()))
                c += 1

    def _construct_hier(self):
        raise NotImplementedError

    def __init__(self, raw_label_path):
        self._raw_labels = self._load_raw_label(raw_label_path)
        self._raw_labels.insert(0, '__background__')
        bk = LabelNode('__background__', 0, True)
        self._label2node = {'__background__': bk}
        self._index2node = [bk]
        self._construct_hier()
        self._compress()
        self._check_dead_node()
        self._raw2path = None

        for node in self._index2node:
            node.info_ratio(self.pos_leaf_sum())
            node.depth_ratio()

        self.max_depth = 0
        for n in self._index2node:
            self.max_depth = max(self.max_depth, n.depth())


