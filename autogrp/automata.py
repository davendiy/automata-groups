#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 21.02.2020
# Excusa. Quod scripsi, scripsi.

# by David Zashkolny
# email: davendiy@gmail.com

from __future__ import annotations

from _autogrp_cython.permutation import Permutation
from _autogrp_cython.tools import lcm, reduce_repetitions, id_func, random_el
from .trees import Tree
from _autogrp_cython.trie import TriedDict

import matplotlib.pyplot as plt
from functools import wraps, partial
import warnings
import typing as tp
from dataclasses import dataclass
from collections import defaultdict, deque
from math import log

REVERSE_VALUE = '@'

AS_WORDS = 1
AS_SHIFTED_WORDS = 1 << 1
AS_GROUP_ELEMENTS = 1 << 2
ONLY_GENERAL = 1 << 3
ALL_FLAGS = 0b1111111111

SKIP_CHILDREN = 1


MAX_ITERATIONS = 10_000


class OutOfGroupError(TypeError):

    def __init__(self, el_name, action):
        super(OutOfGroupError, self).__init__()
        self._name = el_name
        self._action = action

    def __str__(self):
        return f"Can't apply {self._action} to element {self._name} " \
               f"without knowledge of group."


class DifferentGroupsError(TypeError):

    def __init__(self, expected, got):
        super(DifferentGroupsError, self).__init__()
        self.expected = expected
        self.got = got

    def __str__(self):
        return f'Bad group for multiplier: expected {self.expected}, got {self.got}.'


class MaximumOrderDeepError(RecursionError):

    def __init__(self, name):
        super(MaximumOrderDeepError, self).__init__()
        self.name = name

    def __str__(self):
        return f'Reached maximum deep while finding order for {self.name}'


class AutomataTreeNode(Tree):

    _colors = {
        1: 'y',
        2: 'b',
        3: 'r',
        4: 'c',
        5: 'm',
        6: 'g'
    }

    # corresponding labels for colors
    _labels = {
        'b': 'permutation of order 2',
        'r': 'permutation of order 3',
        'c': 'permutation of order 4',
        'm': 'permutation of order 5',
        'g': 'permutation of order 6',
        'y': 'trivial permutation',
        'k': 'reverse node'
    }

    def __init__(self, permutation=None, value=None, reverse=False,
                 simplify=False):
        """ Tree for element of Automata group.

        :param permutation: permutation of node
        :param value: name of element (will be printed on picture)
        :param reverse: True if this node means the recursive callback of
                        the parent
        :param simplify: True if it could be drawn as point with name
                         (e.g. for e, a, b, c)
        """
        self.permutation = permutation
        self.reverse = reverse
        self.simplify = simplify

        self._size = None

        if reverse:
            value = '@'
        elif value is None:
            value = str(permutation)
        super().__init__(value)

    @property
    def name(self):
        return self.value

    def size(self) -> int:
        if self.reverse or self.value == 'e':
            return 0
        if self._size is None:
            self._size = len(self.name)
            for el in self.children:      # type: AutomataTreeNode
                self._size += el.size()
        return self._size

    def get_coords(self, start_x, start_y, scale, deep=0, show_full=False,
                   y_scale_mul=3):
        """ Recursively (DFS) generates sequence of coordinates of vertices for
        drawing the edges.

        :param start_x: x coordinate of the start position on the plane
        :param start_y: y coordinate of the start position on the plane
        :param scale: length of one step of offset.
                      Offset is measure of vertices' displacement relative to
                      left upper corner. Distance between 2 generation == one
                      step of offset.
        :param deep: auxiliary parameter - number of self's generation:param show_full:
        :param y_scale_mul: multiplier of y-axes scale
                            It's used for bigger step between generations
        :param show_full: False if you want elements with attribute 'simplify'
                          to draw as just one node with name.
        :yield: (x, y) - coordinates on the plane
        """
        x_coord = start_x + self._offset * scale
        y_coord = start_y - deep * scale * y_scale_mul
        yield x_coord, y_coord

        if not self.simplify or show_full:
            for child in self.children:  # type: AutomataTreeNode
                for coords in child.get_coords(start_x, start_y, scale,
                                               deep+1, show_full=show_full,
                                               y_scale_mul=y_scale_mul):
                    yield coords
                yield x_coord, y_coord

    def draw(self, start_x=0, start_y=0, scale=10, radius=4, fontsize=50,
             save_filename='', show_full=False, y_scale_mul=3, lbn=False,
             show_names=True):
        """ Draws the tree in matplotlib.

        :param start_x: x coordinate of the start position on the plane
        :param start_y: y coordinate of the start position on the plane
        :param scale: length of one step of offset.
                      Offset is measure of vertices' displacement relative to
                      left upper corner. Distance between 2 generation == one
                      step of offset.
        :param radius: radius of vertices
        :param fontsize: size of font (like fontsize in matplotlib.pyplot.text)
        :param save_filename: name of file, where it should be save.
                              Default == '', means that the picture won't be saved
        :param show_full: False if you want elements with attribute 'simplify'
                          to draw as just one node with name.
        :param y_scale_mul: multiplier of y-axes scale
                            It's used for bigger step between generations
        :param lbn: leaves belong names, True if you want to print names
                    of leaves belong them
        :param show_names: True if you want to plot the names of vertices
        """
        fig, ax = plt.subplots(figsize=(20, 20))
        # ax = fig.add_subplot(111)

        self.make_offsets()    # preparing

        # --------------------------draw edges----------------------------------
        x_coords = []
        y_coords = []
        for x, y in self.get_coords(start_x, start_y, scale, show_full=show_full,
                                    y_scale_mul=y_scale_mul):
            x_coords.append(x)
            y_coords.append(y)

        # fontsize measured in pixels (unlike the radius and another distances
        # on plane), so we need to change its value in order to save the
        # size when matplotlib use autoscale for picture
        fontsize = fontsize / (log(self.vert_amount()))
        ax.plot(x_coords, y_coords, linewidth=0.5)

        # ------------------------draw vertices---------------------------------
        used_colors = set()    # for legend
        self._draw(ax, start_x, start_y, scale, radius, fontsize, show_full=show_full,
                   y_scale_mul=y_scale_mul, used_colors=used_colors, lbn=lbn,
                   show_names=show_names)

        # auxiliary circles with labels
        # used for legend
        for color in used_colors:
            circle = plt.Circle((0, 0), radius=radius/100,
                                color=color, label=self._labels[color])
            ax.add_patch(circle)

        # some kind of deleting of the previous circles
        circle = plt.Circle((0, 0), radius=radius/100, color='w')
        ax.add_patch(circle)

        ax.axis('off')
        ax.set_aspect('equal')
        ax.autoscale_view()
        ax.legend()
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
        fig.show()
        if save_filename:
            fig.savefig(save_filename)

    def _draw(self, ax, start_x, start_y, scale, radius,
              fontsize, deep=0, show_full=False, y_scale_mul=3, used_colors=None,
              lbn=False, show_names=True):
        """ Auxiliary recursive (DFS) function for drawing the vertices of tree.

        :param ax: matplotlib object for plotting
        :param start_x: x coordinate of the start position on the plane
        :param start_y: y coordinate of the start position on the plane
        :param scale: length of one step of offset.
                      Offset is measure of vertices' displacement relative to
                      left upper corner. Distance between 2 generation == one
                      step of offset.
        :param radius: radius of vertices
        :param fontsize: size of font (like fontsize in matplotlib.pyplot.text)
        :param show_full: False if you want elements with attribute 'simplify'
                          to draw as just one node with name.
        :param y_scale_mul: multiplier of y-axes scale
                            it's used for bigger step between generations
        :param used_colors: auxiliary set with all the orders of permutations that
                            were plotted on the plane. It's used for pointing
                            at the legend just colors that we used.
        :param lbn: leaves belong names, True if you want to print names
                    of leaves belong them
        """

        if used_colors is None:
            used_colors = set()

        x_coord = self._offset * scale + start_x
        y_coord = start_y - deep * scale * y_scale_mul

        # plots circles for nodes with different permutations
        # using different colors
        if not self.reverse:
            color = AutomataTreeNode._colors[self.permutation.order()]
            circle = plt.Circle((x_coord, y_coord), radius=radius, color=color)
            used_colors.add(color)
        else:
            # plots reverse nodes using black color
            circle = plt.Circle((x_coord, y_coord), radius=radius, color='k')
            used_colors.add('k')
        ax.add_patch(circle)

        if show_names:
            # bias for printing the name above or belong the node
            # belong is used just for
            if (not self.children or (self.simplify and not show_full)) and lbn:
                bias = -2
            else:
                bias = 1
            ax.annotate(str(self.value), xy=(x_coord, y_coord + radius * bias),
                        xytext=(-fontsize * (len(str(self.value))//3), 2 * bias),
                        textcoords='offset pixels',
                        fontsize=fontsize)

        # continue plotting of children if given parameters allow
        if not self.simplify or show_full:
            for child in self.children:       # type: AutomataTreeNode
                child._draw(ax, start_x, start_y, scale, radius, fontsize,
                            deep + 1, show_full, y_scale_mul, used_colors, lbn,
                            show_names=show_names)

    def add_child(self, child, position=None):
        if isinstance(child, AutomataTreeNode):
            _child = child.copy()       # add only copies of the given trees
        else:                           # in order to avoid cycles
            raise NotImplementedError()
        if position is None:
            self.children.append(_child)
        else:
            self.children.insert(position, _child)
        return _child

    def remove(self, child):
        if isinstance(child, AutomataTreeNode):
            self.children.remove(child)
        else:
            for i, el in enumerate(self.children):
                if el.value == child:
                    del self.children[i]
                    break

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        res = AutomataTreeNode(self.permutation, self.value,
                               self.reverse, self.simplify)
        res._size = self._size

        for child in self.children:   # type: AutomataTreeNode
            res.add_child(child)
        return res


class _Decorators:

    @staticmethod
    def check_group(method):
        @wraps(method)
        def res_method(self, *args, **kwargs):
            if self.parent_group is None:
                raise OutOfGroupError(self.name, method.__name__)
            return method(self, *args, **kwargs)

        return res_method

    @staticmethod
    def cached(attr_name):

        def _cached(method):

            @wraps(method)
            def res_method(self, *args, **kwargs):
                if self.parent_group.cache and getattr(self, attr_name) is not None:
                    return getattr(self, attr_name)
                res = method(self, *args, **kwargs)
                if self.parent_group.cache:
                    setattr(self, attr_name, res)
                return res

            return res_method

        return _cached


@dataclass
class _QueueParams:
    el: AutomataGroupElement
    checked: dict
    last_power: int
    cur_power: int
    deep: int
    check_only: int
    algo: int
    word: str
    predecessor: str


class AutomataGroupElement:

    _order_max_deep = 30

    def __init__(self, name: str, permutation: Permutation,
                 children=None, is_atom=False, group=None):
        self.name = name
        self._perm = permutation

        if children is None:
            self.children = ('e',) * self._perm.size
        else:
            self.children = children

        if len(self.children) != self._perm.size:
            raise ValueError(f"Bad amount of children: expected "
                             f"{self._perm.size}, got {len(self.children)}")

        self.__group = group   # type: AutomataGroup
        self.simplify = is_atom
        self._tree              = None
        self._is_one            = None
        self._order             = None
        self._is_finite         = None
        self._cycle_start_el    = None
        self._cycle_end_el      = None
        self._cycle_len         = None
        self._cycle_start_deep  = None
        self._cycle_end_deep    = None
        self._cycle_start_power = None
        self._cycle_end_power   = None
        self._path_to_cycle     = None

    @classmethod
    def set_order_max_deep(cls, value):
        value = int(value)
        if value < 1:
            raise ValueError(f'Max_deep should be greater than 1')

        cls._order_max_deep = value

    def __len__(self):
        if self.name == 'e':
            return 0
        else:
            return len(self.name)

    @property
    def tree(self) -> AutomataTreeNode:
        self._calc_tree()
        return self._tree

    @property
    def permutation(self) -> Permutation:
        return self._perm

    @property
    def cardinality(self) -> int:
        return len(self.children)

    @property
    def parent_group(self):
        return self.__group

    @parent_group.setter
    def parent_group(self, value):
        if not isinstance(value, AutomataGroup):
            raise TypeError(f"Parent should be instance of Automata group, not {type(value)}")
        self.__group = value
        self._is_one = None
        self._order = None
        self._is_finite = None

    @_Decorators.check_group
    def __iter__(self) -> AutomataGroupElement:
        for el in self.children:
            yield self.parent_group(el)

    @_Decorators.check_group
    def dfs(self):
        yield self
        for child in self:    # type: AutomataGroupElement
            if child.name == self.name:
                continue
            if child.simplify:
                yield child
                continue

            for el in child.dfs():
                yield el

    @_Decorators.check_group
    def bfs(self):
        queue = deque([self])
        while queue:
            cur_el = queue.pop()
            yield cur_el
            if not cur_el.simplify:
                queue.extend(el for el in cur_el)

    @_Decorators.check_group
    def __getitem__(self, item):
        if isinstance(item, int) or \
                str(item.__class__) == "<class 'sage.rings.integer.Integer'>":
            return self.parent_group(self.children[item])
        elif isinstance(item, tuple):
            res = self
            for el in item:
                res = self.parent_group(res[el])
            return res
        else:
            raise TypeError(f"Tree indices must be int or tuple, not {type(item)}")

    @_Decorators.check_group
    def __call__(self, word) -> str:
        if isinstance(word, int) or \
                str(word.__class__) == "<class 'sage.rings.integer.Integer'>":
            word = str(word)

        if not isinstance(word, str):
            raise TypeError(f'Word should be int or str, not {type(word)}')
        elif not (got := set(word)) <= self.parent_group.expected_words:
            raise ValueError(f'Bad elements for applying group action: '
                             f'{got - self.parent_group.expected_words}')

        now_el = self
        res = ''
        for el in word:
            el = int(el)
            res += str(now_el.permutation(el))
            now_el = now_el[el]
        return res

    @_Decorators.check_group
    def call_graph(self, word, graph):
        word = str(word)
        prev_word = word
        x = self(word)
        graph.add_vertex(word)
        checked = set()
        i = 0
        while True:
            graph.add_edge(prev_word, x)
            if x in checked or i > MAX_ITERATIONS:
                break

            prev_word = x
            x = self(x)
            i += 1

    def __repr__(self):
        return f'{self.parent_group.name}({self.name} = {self.permutation} ({", ".join(self.children)}))'

    @_Decorators.check_group
    def __mul__(self, other):
        if not isinstance(other, AutomataGroupElement):
            raise TypeError(f"Multiplier should be instance of AutomataGroupElement, not {type(other)}")
        if self.parent_group != other.parent_group:
            raise DifferentGroupsError(self.parent_group, other.parent_group)
        return self.parent_group.multiply(self, other)

    @_Decorators.check_group
    def __pow__(self, power):
        if not (isinstance(power, int) or
                str(power.__class__) == "<class 'sage.rings.integer.Integer'>"):
            raise TypeError(f"Power type should be int, not {type(power)}")

        if power == -1:
            return self.parent_group(self.name[::-1])

        if self.parent_group.is_defined(self.name * power):
            return self.parent_group(self.name * power)

        res = self.parent_group.one
        tmp = self
        i = 1
        while i <= power:
            if i & power:
                res *= tmp
            i <<= 1
            tmp *= tmp
        return res

    @_Decorators.check_group
    def inverse(self):
        return self ** (-1)

    @_Decorators.check_group
    @_Decorators.cached('_is_one')
    def is_one(self):

        if self.name == 'e':
            res = True
        elif self.permutation.order() != 1:
            res = False
        else:
            queue = set()
            checked = set()
            queue.update(el for el in self.children if el != self.name)
            res = True
            while queue:
                tmp = queue.pop()
                checked.add(tmp)
                child = self.parent_group(tmp)

                if child.permutation.order() != 1:
                    res = False
                    break
                queue.update(el for el in child.children if el not in checked)

        return res

    @_Decorators.check_group
    @_Decorators.cached('_order')
    def order(self, check_finite=True):

        if self.is_one():
            res = 1
        elif check_finite and not self.is_finite():
            res = float('inf')
        else:
            power = int(self.permutation.order())
            next_el = self ** power
            res = 1
            for el in next_el:    # type: AutomataGroupElement
                res = lcm(res, el.order(check_finite=False))
            res *= power
        return res

    @_Decorators.check_group
    @_Decorators.cached('_is_finite')
    def is_finite(self, check_only=None, verbose=False, use_dfs=False,
                  algo=AS_SHIFTED_WORDS, print_full_els=False):

        if (algo & ONLY_GENERAL) and check_only is not None:
            raise ValueError(f"Can't apply simultaneously "
                             f"check_only={check_only} and ONLY_GENERAL flags")

        old_cache = self.parent_group.cache
        if check_only is not None and old_cache:
            warnings.warn(f"Used check_only={check_only} with enabled caching. "
                          f"Disabling caching...")
            self.parent_group.disable_cache()

        if use_dfs:
            res = self._is_finite_dfs(check_only=check_only, verbose=verbose,
                                      algo=algo,
                                      print_full_els=print_full_els)
        else:
            res = self._is_finite_bfs(check_only=check_only, verbose=verbose,
                                      algo=algo,
                                      print_full_els=print_full_els)
        if check_only is not None and old_cache:
            warnings.warn(f"Enabling cache again.")
            self.parent_group.enable_cache()
        return res

    @_Decorators.cached('_is_finite')
    def _is_finite_dfs(self, cur_power=1, algo=AS_SHIFTED_WORDS, checked=None,
                       check_only=None, deep=0,
                       verbose=False, print_full_els=False):
        if verbose:
            printed = str(self) if print_full_els else self.name
            print(f"Entered {deep} generation. Element: {printed}")
        if deep > self._order_max_deep:
            raise MaximumOrderDeepError(self.name)
        if checked is None:
            checked = {}

        res = True

        if not self.is_one():

            # check whether any of cyclic shifts of name was checked
            # elements with same names up to cycle shifts are conjugate
            # and therefore have same order
            found = self._find_in_checked(self.name, checked, algo)
            prev_power = checked.get(found)
            if found and prev_power != cur_power:
                if verbose:
                    print(f'Found cycle between {self.name} and {found} '
                          f'of length {cur_power / prev_power}')
                return False
            elif found:
                return True
            checked[self.name] = cur_power

            orbits = self._get_orbits(self.permutation, algo)

            # remove flag ONLY_GENERAL from next usages
            nex_algo = algo & (ALL_FLAGS ^ ONLY_GENERAL)

            # always start from orbit of 0
            for orbit in sorted(orbits):
                if check_only is not None and check_only not in orbit:
                    continue

                # next generation will check only specified element
                if algo & ONLY_GENERAL:
                    check_only = min(orbit)
                power = len(orbit)
                next_el = (self ** power)[orbit[0]]

                if not next_el._is_finite_dfs(cur_power=cur_power * power,
                                              checked=checked, algo=nex_algo,
                                              check_only=check_only,
                                              deep=deep+1, verbose=verbose):
                    res = False
                    break
            del checked[self.name]

        return res

    def _is_finite_bfs(self, check_only=None,
                       verbose=False, algo=AS_SHIFTED_WORDS,
                       print_full_els=False):

        for params in (order_gen := self._order_bfs(check_only, algo)):

            printed = str(params.el) if print_full_els else params.el.name
            if verbose:
                print(f'Generation: {params.deep}, element: {printed}')
            if params.el.is_one():
                order_gen.send(SKIP_CHILDREN)
                continue

            if params.deep > self._order_max_deep:
                raise MaximumOrderDeepError(self.name)

            found = self._find_in_checked(params.el.name, params.checked, params.algo)

            prev_power, prev_deep = params.checked.get(found, (1, 1))

            if found and prev_power != params.cur_power:
                if verbose:
                    print(f'Found cycle between {params.el.name} and {found} '
                          f'of length {params.cur_power / prev_power}')

                self._cycle_start_el = found
                self._cycle_end_el = params.el.name
                self._cycle_start_deep = prev_deep
                self._cycle_end_deep = params.deep
                self._cycle_start_power = prev_power
                self._cycle_end_power = params.cur_power
                self._cycle_len = params.cur_power / prev_power
                self._path_to_cycle = params.word
                # if we found cycle of non-unitary length it means that
                # all of predecessors are infinite elements

                for prev in params.checked:
                    prev = self.parent_group(prev)
                    if prev._is_finite is None and self.parent_group.cache:
                        prev._is_finite = False
                    elif self.parent_group.cache:
                        assert not prev._is_finite, f'Better check {prev}'

                return False

            elif found:
                if verbose:
                    print(f'Found cycle on {params.el.name} '
                          f'of length {params.cur_power / prev_power}')
                order_gen.send(SKIP_CHILDREN)
                continue

        return True

    @_Decorators.check_group
    def order_bfs(self, check_only=None, algo=AS_SHIFTED_WORDS, max_deep=10):
        for params in (gen := self._order_bfs(check_only=check_only, algo=algo)):
            yield params
            if params.el.is_one() or params.deep >= max_deep:
                gen.send(SKIP_CHILDREN)
            found = self._find_in_checked(params.el.name, params.checked, params.algo)
            if found:
                gen.send(SKIP_CHILDREN)

    def _order_bfs(self, check_only=None, algo=AS_SHIFTED_WORDS):
        queue = deque()
        queue.append(_QueueParams(el=self, checked={},
                                  cur_power=1, deep=0, predecessor='',
                                  check_only=check_only, algo=algo, word='',
                                  last_power=1))
        while queue:

            params = queue.popleft()    # type: _QueueParams
            flag = (yield params)

            if flag == SKIP_CHILDREN:
                yield
                continue

            params.checked[params.el.name] = params.cur_power, params.deep

            # remove flag ONLY_GENERAL from next usages
            nex_algo = params.algo & (ALL_FLAGS ^ ONLY_GENERAL)
            orbits = self._get_orbits(params.el.permutation, params.algo)
            for orbit in sorted(orbits):
                if params.check_only is not None and params.check_only not in orbit:
                    continue
                if params.algo & ONLY_GENERAL:
                    next_check_only = min(orbit)
                    power = params.el.permutation.order()
                else:
                    next_check_only = params.check_only
                    power = len(orbit)
                next_el = (params.el ** power)[orbit[0]]
                queue.append(_QueueParams(
                    el=next_el, checked=params.checked.copy(),
                    cur_power=params.cur_power * power, deep=params.deep + 1,
                    algo=nex_algo, check_only=next_check_only, word=params.word + str(orbit[0]),
                    predecessor=params.el.name, last_power=power)
                )

    @staticmethod
    def _get_orbits(permutation, algo=0):

        # we don't use orbits when apply ONLY_GENERAL algo
        # because that won't work on abcfc \in H4, since
        # its permutation is cycle of length 4 and 0-branch has
        # infinite set of elements
        if algo & ONLY_GENERAL:
            return [[i] for i in range(permutation.size)]
        else:
            fixed_points = [[i] for i in range(permutation.size)
                            if permutation(i) == i]
            cycles = permutation.cyclic_form
            return fixed_points + cycles

    def _find_in_checked(self, el_name: str, checked, algo):
        if algo & AS_WORDS:
            return el_name if el_name in checked else ''

        # check whether any of cyclic shifts of name was checked
        # elements with same names up to cycle shifts are conjugate
        # and therefore have same order
        elif algo & AS_SHIFTED_WORDS:
            for i in range(len(el_name)):
                tmp_name = el_name[i:] + el_name[:i]
                if tmp_name in checked:
                    return tmp_name
            return ''

        # a == b as elements of the group <=> a * b^(-1) - trivial element
        elif algo & AS_GROUP_ELEMENTS:
            for prev in checked:
                tmp = self.parent_group(prev) \
                      * self.parent_group(el_name).inverse()
                if tmp.is_one():
                    return prev
            return ''
        else:
            raise ValueError(f"Unknown algorithm flag: {algo}.")

    @_Decorators.check_group
    def order_graph(self, graph, loops=False, as_tree=False, max_deep=10,
                    short_names=True, algo=AS_SHIFTED_WORDS):
        added = defaultdict(int)
        checked = {}
        added[self.name] += 1
        cur_vertex = self._create_name(self.name, 'START', short_names)
        graph.add_vertex(cur_vertex)
        checked[self.name] = cur_vertex
        self._order_graph(graph=graph, checked=checked, added=added,
                          loops=loops, as_tree=as_tree, deep=0,
                          max_deep=max_deep, short_names=short_names,
                          algo=algo)

    def _order_graph(self, graph, checked, added, loops, as_tree, deep=0,
                     max_deep=10, short_names=True, algo=AS_SHIFTED_WORDS):
        cur_vertex = checked[self.name]
        if self.is_one():
            return

        if deep > max_deep:
            return
        if deep > self._order_max_deep:
            raise MaximumOrderDeepError(self.name)

        orbits = self._get_orbits(self.permutation)
        for orbit in sorted(orbits):
            power = len(orbit)
            next_el = (self ** power)[orbit[0]]

            found = self._find_in_checked(next_el.name, checked, algo)

            if not loops and found == self.name:
                continue

            if found:
                if as_tree:
                    continue
                dest = checked[found]
                graph.add_edge(cur_vertex, dest, f'^{power}|{orbit[0]}')
            else:
                added[next_el.name] += 1
                next_vertex = self._create_name(next_el.name,
                                                added[next_el.name], short_names)
                graph.add_vertex(next_vertex)
                checked[next_el.name] = next_vertex
                graph.add_edge(cur_vertex, next_vertex, f'^{power}|{orbit[0]}')
                next_el._order_graph(graph=graph, checked=checked,
                                     added=added, loops=loops, as_tree=as_tree,
                                     deep=deep+1, max_deep=max_deep,
                                     short_names=short_names, algo=algo)
                del checked[next_el.name]

    # TODO: add unit tests
    @staticmethod
    def _create_name(name, number, short_names=True):
        if short_names and len(name) > 7:
            return f'{name[:3]}<{len(name) - 6}>{name[-3:]} # {number}'
        else:
            return f'{name} # {number}'

    @_Decorators.check_group
    def _calc_tree(self):
        if self._tree is not None:
            return self._tree
        self._tree = AutomataTreeNode(self.permutation,
                                      self.name,
                                      reverse=False,
                                      simplify=self.simplify)

        for el in self:    # type: AutomataGroupElement
            if el.name == self.name:
                child = AutomataTreeNode(reverse=True)
            else:
                child = el._calc_tree()
            self._tree.add_child(child)
        return self._tree

    @property
    def word_with_cycle_orbit(self):
        if self._path_to_cycle is None:
            return None

        res_word = self._path_to_cycle[:self._cycle_start_deep] + '(' \
                   + self._path_to_cycle[self._cycle_start_deep:] + ')'
        return res_word

    # TODO: add unit tests
    def describe(self, graph_class=None, show_structure=True,
                 y_scale_mul=5, max_deep=7, loops=True,
                 figsize=(15, 15), vertex_size=15, print_full_els=False,
                 verbose=True, short_names=True, as_tree=False, algo=AS_SHIFTED_WORDS):
        old = self.parent_group.cache
        self.parent_group.disable_cache()
        n = max(100 - len(str(self)), 0)
        n //= 2
        tmp = (
            f"{'=' * n}{str(self)}{'=' * n}\n"
            f'Group:     {self.parent_group}\n'
            f'size:      {self.tree.size()}\n'
            f'height:    {self.tree.height()}\n'
        )
        print(tmp)

        is_finite = self.is_finite(verbose=verbose,
                                   print_full_els=print_full_els,
                                   algo=algo)
        tmp = (
            f'\nis finite: {is_finite}\n'
            f"order:     {self.order(check_finite=False) if is_finite else 'inf'}\n\n"
        )

        tmp2 = (
            f'Found cycle\n'
            f'    start deep:   {self._cycle_start_deep}\n'
            f'    end deep:     {self._cycle_end_deep}\n'
            f'    start el:     {self._cycle_start_el}\n'
            f'    end el:       {self._cycle_end_el}\n'
            f'    start power:  {self._cycle_start_power}\n'
            f'    end power:    {self._cycle_end_power}\n'
            f'    cycle weight: {self._cycle_len}\n'
            f'    full path:    {self._path_to_cycle}\n'
            f'    word with cycle orbit:  {self.word_with_cycle_orbit}\n'
            f"{'=' * 100}\n"
        )
        if old:
            self.parent_group.enable_cache()

        if self._cycle_start_el is not None:
            tmp += tmp2
        print(tmp)

        if show_structure:
            self.show(y_scale_mul=y_scale_mul)
        if graph_class is not None:
            graph = graph_class(loops=loops, multiedges=True)
            self.order_graph(graph, max_deep=max_deep, loops=loops,
                             short_names=short_names, as_tree=as_tree, algo=algo)
            _plt = graph.plot(edge_labels=True, vertex_size=vertex_size, edge_style=':')
            _plt.show(figsize=figsize)

    def show(self, start_x=0, start_y=0, scale=10, radius=4, fontsize=50,
             save_filename='', show_full=False, y_scale_mul=3, lbn=False,
             show_names=True):
        self.tree.draw(start_x=start_x, start_y=start_y,
                       scale=scale, radius=radius,
                       fontsize=fontsize, y_scale_mul=y_scale_mul,
                       save_filename=save_filename, show_full=show_full,
                       lbn=lbn, show_names=show_names)


class AutomataGroup:

    __instances = {}

    @classmethod
    def all_instances(cls):
        return cls.__instances

    def __new__(cls, name, gens: tp.List[AutomataGroupElement],
                reduce_function=id_func, lempel_ziv=True):
        if not gens:
            raise ValueError("Gens should be a non-empty list of AutomataGroupElement-s")

        if name not in cls.__instances:
            obj = super(AutomataGroup, cls).__new__(cls)
            obj.name = name
            obj._defined_els = TriedDict()
            obj._defined_trees = {}
            obj._lempel_ziv = lempel_ziv
            obj._use_cache = True

            obj.__gens = gens
            obj._size = gens[-1].permutation.size
            obj._expected_words = set(str(el) for el in range(obj._size))

            obj._e = AutomataGroupElement('e', Permutation(obj._size),
                                          group=obj, is_atom=True)
            obj._reduce_func = reduce_function
            obj._defined_els['e'] = obj._e
            for el in obj.__gens:
                obj._defined_els[el.name] = el
                el.parent_group = obj
            cls.__instances[name] = obj
        else:
            warnings.warn(f"There is already group with name {name}, so creation "
                          f"of new such group just returns previous one and "
                          f"doesn't change its attributes. "
                          f"Use AutomataGroup.clear_group('{name}') before.")

        return cls.__instances[name]

    @property
    def expected_words(self):
        return self._expected_words

    def enable_cache(self):
        self._use_cache = True

    def disable_cache(self):
        self._use_cache = False

    @property
    def cache(self):
        return self._use_cache

    def enable_lempel_ziv(self):
        if not self._lempel_ziv:
            self.clear_memory()
            self._lempel_ziv = True

    def disable_lempel_ziv(self):
        if self._lempel_ziv:
            self.clear_memory()
            self._lempel_ziv = False

    @classmethod
    def clear_group(cls, name):
        if name not in cls.__instances:
            raise KeyError(f'There is no group with name {name}.')
        del cls.__instances[name]

    @property
    def alphabet(self):
        return [el.name for el in self.__gens]

    def clear_memory(self):
        self._defined_els.clear()
        self._defined_els['e'] = self._e
        for el in self.__gens:
            self._defined_els[el.name] = el

    def is_defined(self, el) -> bool:
        if not isinstance(el, str):
            raise TypeError(f"el should be str, not {type(el)}")
        return el in self._defined_els

    @property
    def one(self) -> AutomataGroupElement:
        return self._e

    def random_el(self, length, allow_same_neighbours=False):
        return self(random_el(self.alphabet, repeat=length,
                              allow_same_neighbours=allow_same_neighbours))

    @property
    def gens(self) -> tp.List[AutomataGroupElement]:
        return self.__gens

    @property
    def size(self):
        return self._size

    def __call__(self, word) -> AutomataGroupElement:
        if isinstance(word, str):
            if word not in self._defined_els:     # Lempel-Ziv-like algorithm
                res = self._e
                if self._lempel_ziv:
                    left_word = word
                    last_value = self._e
                    while left_word:
                        prefix, left_word, value = self._defined_els.max_prefix(left_word)
                        if prefix not in self._defined_els:
                            raise ValueError(f"Unknown prefix: {prefix}")
                        res *= value
                        _ = last_value * value     # cache last two prefixes
                        last_value = value
                else:
                    for el in word:
                        if el not in self._defined_els:
                            raise ValueError(f'Unknown element: {el}')
                        res *= self._defined_els[el]

                self._defined_els[word] = res
            return self._defined_els[word]
        elif isinstance(word, AutomataGroupElement):
            return self.__call__(word.name)
        else:
            raise TypeError(f"Not supported type: {type(word)}")

    def multiply(self, x1, x2) -> AutomataGroupElement:

        if x1.name == 'e':
            return x2
        if x2.name == 'e':
            return x1

        res_name = self._reduce_func(x1.name + x2.name)
        res_name = res_name if res_name else 'e'
        if not self.is_defined(res_name):

            res_permutation = x2.permutation * x1.permutation  # other(self())
            res_children = []

            for i in range(self.size):
                a_child = x1.children[x2.permutation(i)]
                b_child = x2.children[i]
                child_name = self._reduce_func(a_child + b_child)
                child_name = child_name.replace('e', '')
                res_children.append(child_name if child_name else 'e')

            self._defined_els[res_name] = \
                AutomataGroupElement(res_name, res_permutation, tuple(res_children), group=self)
        return self._defined_els[res_name]

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        generators = ', '.join(str(el) for el in self.__gens)
        text = f"""
        \rAutomataGroup {self.name}
        \rover alphabet {self._expected_words}
        \rgenerated by <{generators}>.
        """
        return text

    @classmethod
    def generate_H3(cls, apply_reduce_func=False, force=False):
        _a = AutomataGroupElement('a', permutation=Permutation([1, 0, 2]),
                                  children=('e', 'e', 'a'), is_atom=True)
        _b = AutomataGroupElement('b', permutation=Permutation([2, 1, 0]),
                                  children=('e', 'b', 'e'), is_atom=True)
        _c = AutomataGroupElement('c', permutation=Permutation([0, 2, 1]),
                                  children=('c', 'e', 'e'), is_atom=True)

        if force and 'H3' in cls.all_instances():
            cls.clear_group('H3')
        if apply_reduce_func:
            reduce_func = partial(reduce_repetitions, atoms=['a', 'b', 'c'])
            return AutomataGroup('H3', [_a, _b, _c], reduce_function=reduce_func)
        else:
            return AutomataGroup('H3', [_a, _b, _c])

    @classmethod
    def generate_H3_conjugated(cls, apply_reduce_func=False, force=False):
        _a = AutomataGroupElement('a', permutation=Permutation([1, 0, 2]),
                                  children=('e', 'e', 'b'), is_atom=True)
        _b = AutomataGroupElement('b', permutation=Permutation([2, 1, 0]),
                                  children=('e', 'a', 'e'), is_atom=True)
        _c = AutomataGroupElement('c', permutation=Permutation([0, 2, 1]),
                                  children=('c', 'e', 'e'), is_atom=True)

        if force and 'H3' in cls.all_instances():
            cls.clear_group('H3')
        if apply_reduce_func:
            reduce_func = partial(reduce_repetitions, atoms=['a', 'b', 'c'])
            return AutomataGroup('H3*', [_a, _b, _c], reduce_function=reduce_func)
        else:
            return AutomataGroup('H3*', [_a, _b, _c])

    @classmethod
    def generate_H4(cls, apply_reduce_func=False, force=False):
        _a = AutomataGroupElement('a', permutation=Permutation([0, 2, 1, 3]),
                                  children=('a', 'e', 'e', 'a'),
                                  is_atom=True)
        _b = AutomataGroupElement('b', permutation=Permutation([2, 1, 0, 3]),
                                  children=('e', 'b', 'e', 'b'),
                                  is_atom=True)
        _c = AutomataGroupElement('c', permutation=Permutation([1, 0, 2, 3]),
                                  children=('e', 'e', 'c', 'c'),
                                  is_atom=True)
        _d = AutomataGroupElement('d', permutation=Permutation([0, 1, 3, 2]),
                                  children=('d', 'd', 'e', 'e'),
                                  is_atom=True)
        _f = AutomataGroupElement('f', permutation=Permutation([0, 3, 2, 1]),
                                  children=('f', 'e', 'f', 'e'),
                                  is_atom=True)
        _g = AutomataGroupElement('g', permutation=Permutation([3, 1, 2, 0]),
                                  children=('e', 'g', 'g', 'e'),
                                  is_atom=True)
        if force and 'H4' in cls.all_instances():
            cls.clear_group('H4')

        if apply_reduce_func:
            reduce_func = partial(reduce_repetitions, atoms=list('abcdfg'))
            return AutomataGroup('H4', [_a, _b, _c, _d, _f, _g],
                                 reduce_function=reduce_func)
        else:
            return AutomataGroup('H4', [_a, _b, _c, _d, _f, _g])
