#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 21.02.2020
# Excusa. Quod scripsi, scripsi.

# by David Zashkolny
# email: davendiy@gmail.com


from .permutation import Permutation
from .tools import lcm

from functools import wraps
import warnings
import typing as tp
from collections import defaultdict

ORDER_MAX_DEEP = 30
REVERSE_VALUE = '@'


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


class SingletonError(ValueError):

    def __init__(self, name):
        super(SingletonError, self).__init__()
        self.name = name

    def __str__(self):
        return f'There is already group with name {self.name}.'


class MaximumOrderDeepError(RecursionError):

    def __init__(self, name):
        super(MaximumOrderDeepError, self).__init__()
        self.name = name

    def __str__(self):
        return f'Reached maximum deep while finding order for {self.name}'


def _check_group(method):
    @wraps(method)
    def res_method(self, *args, **kwargs):
        if self.parent_group is None:
            raise OutOfGroupError(self.name, method.__name__)
        return method(self, *args, **kwargs)

    return res_method


class AutomataGroupElement:

    def __init__(self, name, permutation,
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
        self._tree = None
        self._is_one = None
        self._order = None
        self._is_finite = None

    def __len__(self):
        if self.name == 'e':
            return 0
        else:
            return len(self.name)

    @property
    def tree(self):
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

    @_check_group
    def __iter__(self):
        for el in self.children:
            yield self.parent_group(el)

    @_check_group
    def __getitem__(self, item):
        if isinstance(item, int):
            return self.parent_group(self.children[item])
        elif isinstance(item, tuple):
            res = self
            for el in item:
                res = self.parent_group(res[el])
            return res
        else:
            raise TypeError(f"Tree indices must be int or tuple, not {type(item)}")

    def __repr__(self):
        return f'{self.parent_group.name}({self.name} = {self.permutation} ({", ".join(self.children)}))'

    def __str__(self):
        return self.__repr__()

    @_check_group
    def __mul__(self, other):
        if not isinstance(other, AutomataGroupElement):
            raise TypeError("Bad type for multiplier.")
        if self.parent_group != other.parent_group:
            raise DifferentGroupsError(self.parent_group, other.parent_group)
        return self.parent_group.multiply(self, other)

    @_check_group
    def __pow__(self, power):
        if not isinstance(power, int):
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

    @_check_group
    def is_one(self, use_cache=True):

        if use_cache and self._is_one is not None:
            return self._is_one

        if self.name == 'e':
            self._is_one = True
        elif self.permutation.order() != 1:
            self._is_one = False
        else:
            queue = set()
            queue.update(el for el in self.children if el != self.name)
            res = True
            while queue:
                tmp = queue.pop()
                child = self.parent_group(tmp)

                if child.permutation.order() != 1:
                    res = False
                    break
                queue.update(el for el in child.children if el != child.name)
            self._is_one = res
        return self._is_one

    @_check_group
    def order(self, check_finite=True, use_cache=True):
        if use_cache and self._order is not None:
            return self._order

        if self.is_one():
            self._order = 1
        elif check_finite and not self.is_finite():
            self._order = float('inf')
        else:
            power = int(self.permutation.order())
            next_el = self ** power
            res = 1
            for el in next_el:    # type: AutomataGroupElement
                res = lcm(res, el.order(check_finite=False, use_cache=use_cache))
            self._order = res * power
        return self._order

    @_check_group
    def is_finite(self, cur_power=1, checked=None,
                        check_only_0=False, deep=1, verbose=False, use_cache=True):

        if use_cache and self._is_finite is not None:
            return self._is_finite

        if verbose:
            print(f"Entered {deep} generation. Name: {self.name}")
        if deep > ORDER_MAX_DEEP:
            raise MaximumOrderDeepError(self.name)
        if checked is None:
            checked = {}

        self._is_finite = True

        if not self.is_one():

            # check whether any of cyclic shifts of name was checked
            # elements with same names up to cycle shifts are conjugate
            # and therefore have same order
            for i in range(len(self.name)):
                tmp_name = self.name[i:] + self.name[:i]
                if tmp_name in checked:
                    if verbose:
                        print('found cycle:', self.name, tmp_name)
                    prev_power = checked[tmp_name]
                    return prev_power == cur_power

            checked[self.name] = cur_power

            # permutation.cyclic_form doesn't return cycles of unitary length
            # (i.e. fixed points)
            fixed_points = [[i] for i in range(self.cardinality)
                                        if self.permutation(i) == i]
            cycles = self.permutation.cyclic_form
            orbits = fixed_points + cycles

            # FIXME rewrite using BFS
            # always start from orbit of 0
            for orbit in sorted(orbits):
                power = len(orbit)
                next_el = (self ** power)[orbit[0]]
                if not next_el.is_finite(cur_power=cur_power * power, checked=checked,
                                         check_only_0=check_only_0,
                                         deep=deep+1, verbose=verbose,
                                         use_cache=use_cache):
                    self._is_finite = False
                    break

                if check_only_0:
                    break
            del checked[self.name]

        return self._is_finite

    @_check_group
    def order_graph(self, graph, loops=False, as_tree=False):
        added = defaultdict(int)
        checked = {}
        added[self.name] += 1
        cur_vertex = f'{self.name} #{added[self.name]}'
        graph.add_vertex(cur_vertex)
        checked[self.name] = cur_vertex
        self._order_graph(graph, checked, added, loops, as_tree)

    def _order_graph(self, graph, checked, added, loops, as_tree):
        cur_vertex = checked[self.name]
        if self.is_one():
            return

        fixed_points = [[i] for i in range(self.cardinality)
                                    if self.permutation(i) == i]
        cycles = self.permutation.cyclic_form
        orbits = fixed_points + cycles
        for orbit in orbits:
            power = len(orbit)
            next_el = (self ** power)[orbit[0]]
            next_el_name = next_el.name
            if not loops and next_el_name == self.name:
                continue

            if next_el_name in checked:
                if as_tree:
                    continue
                dest = checked[next_el_name]
                graph.add_edge(cur_vertex, dest, power)
            else:
                added[next_el_name] += 1
                next_vertex = f'{next_el_name} #{added[next_el_name]}'
                graph.add_vertex(next_vertex)
                checked[next_el_name] = next_vertex
                graph.add_edge(cur_vertex, next_vertex, power)
                next_el._order_graph(graph, checked, added, loops, as_tree)
                del checked[next_el.name]

    # @check_group
    # def _calc_tree(self):
    #     if self.tree is not None:
    #         return self.tree
    #
    #     if self.simplify:
    #
    #
    #     children = []
    #     for el in self:    # type: AutomataGroupElement
    #         if el.name == self.name:
    #             children.append()
    #         children.append(el._calc_tree())


class AutomataGroup:

    __instances = {}

    @classmethod
    def all_instances(cls):
        return cls.__instances

    def __new__(cls, name, gens: tp.List[AutomataGroupElement]):
        if not gens:
            raise ValueError("Gens should be a non-empty list of AutomataGroupElement-s")

        if name not in cls.__instances:
            obj = super(AutomataGroup, cls).__new__(cls)
            obj.name = name
            obj._defined_els = {}
            obj._defined_trees = {}

            obj.__gens = gens
            obj._size = gens[-1].permutation.size
            obj._e = AutomataGroupElement('e', Permutation(obj._size), group=obj)
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

    @classmethod
    def clear_group(cls, name):
        if name not in cls.__instances:
            raise KeyError(f'There is no group with name {name}.')
        del cls.__instances[name]

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

    @property
    def gens(self) -> tp.List[AutomataGroupElement]:
        return self.__gens

    @property
    def size(self):
        return self._size

    # def create_tree_node(self, permutation, value=None, reverse=False,
    #                      simplify=False):

    def __call__(self, el) -> AutomataGroupElement:
        if isinstance(el, str):
            if el not in self._defined_els:
                res = self._e
                for prim_el in el:
                    if prim_el not in self._defined_els:
                        raise ValueError(f"Unknown element: {prim_el}")
                    res *= self._defined_els[prim_el]
                self._defined_els[el] = res
            return self._defined_els[el]
        elif isinstance(el, AutomataGroupElement):
            return self.__call__(el.name)
        else:
            raise TypeError(f"Not supported type: {type(el)}")

    def multiply(self, x1, x2):

        if x1.name == 'e':
            return x2
        if x2.name == 'e':
            return self

        res_name = x1.name + x2.name
        if not self.is_defined(res_name):

            res_permutation = x2.permutation * x1.permutation  # other(self())
            res_children = []

            for i in range(self.size):
                a_child = x1[i ^ x2.permutation]
                b_child = x2[i]
                child_name = a_child.name + b_child.name
                child_name = child_name.replace('e', '')
                res_children.append(child_name if child_name else 'e')

            self._defined_els[res_name] = \
                AutomataGroupElement(res_name, res_permutation, tuple(res_children), group=self)
        return self._defined_els[res_name]

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return self.name

    @staticmethod
    def generate_H3():
        _a = AutomataGroupElement('a', permutation=Permutation([1, 0, 2]),
                                  children=('e', 'e', 'a'), is_atom=True)
        _b = AutomataGroupElement('b', permutation=Permutation([2, 1, 0]),
                                  children=('e', 'b', 'e'), is_atom=True)
        _c = AutomataGroupElement('c', permutation=Permutation([0, 2, 1]),
                                  children=('c', 'e', 'e'), is_atom=True)
        return AutomataGroup('H3', [_a, _b, _c])

    @staticmethod
    def generate_H4():
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
        return AutomataGroup("H4", [_a, _b, _c, _d, _f, _g])
