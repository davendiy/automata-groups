#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 21.02.2020
# Excusa. Quod scripsi, scripsi.

# by David Zashkolny
# email: davendiy@gmail.com


from .permutation import Permutation
from .tools import lcm, reduce_repetitions, id_func
from .trees import Tree
from .trie import TriedDict

import matplotlib.pyplot as plt
from functools import wraps, partial
import warnings
import typing as tp
from collections import defaultdict, deque
from math import log

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


class AutomataTreeNode(Tree):

    # color of the vertex depends on its permutation
    # could be changed into 'unique color for each permutation'
    # now color is unique for each possible order of permutation
    # _colors = {
    #     '(2)': 'y',
    #     '(0 1 2)': 'r',
    #     '(0 2 1)': 'r',
    #     '(2)(0 1)': 'b'
    #     '(0 2)': 'b',
    #     '(1 2)': 'b'
    # }
    #

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
                if self._use_cache and getattr(self, attr_name) is not None:
                    return getattr(self, attr_name)
                else:
                    res = method(self, *args, **kwargs)
                    setattr(self, attr_name, res)
                    return res

            return res_method

        return _cached


class AutomataGroupElement:

    _order_max_deep = 30
    _use_cache = True

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
        self._cycle_start_el = None
        self._cycle_end_el = None
        self._cycle_len = None
        self._cycle_start_deep = None
        self._cycle_end_deep = None
        self._cycle_start_power = None
        self._cycle_end_power = None

    @classmethod
    def set_order_max_deep(cls, value):

        if not isinstance(value, int):
            raise TypeError(f"Max_deep should be int, not {type(value)}")

        if value < 1:
            raise ValueError(f'Max_deep should be greater than 1')

        cls._order_max_deep = value

    @classmethod
    def enable_cache(cls):
        cls._use_cache = True

    @classmethod
    def disable_cache(cls):
        cls._use_cache = False

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
    def __iter__(self):
        for el in self.children:
            yield self.parent_group(el)

    @_Decorators.check_group
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

    @_Decorators.check_group
    def __mul__(self, other):
        if not isinstance(other, AutomataGroupElement):
            raise TypeError(f"Multiplier should be instance of AutomataGroupElement, not {type(other)}")
        if self.parent_group != other.parent_group:
            raise DifferentGroupsError(self.parent_group, other.parent_group)
        return self.parent_group.multiply(self, other)

    @_Decorators.check_group
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

    @_Decorators.check_group
    @_Decorators.cached('_is_one')
    def is_one(self):

        if self.name == 'e':
            res = True
        elif self.permutation.order() != 1:
            res = False
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
    def is_finite(self, check_only=None, verbose=False, use_dfs=False):
        if use_dfs:
            return self._is_finite_dfs(check_only=check_only, verbose=verbose)
        else:
            return self._is_finite_bfs(check_only=check_only, verbose=verbose)

    @_Decorators.cached('_is_finite')
    def _is_finite_dfs(self, cur_power=1, checked=None,
                       check_only=None, deep=1,
                       verbose=False):
        if verbose:
            print(f"Entered {deep} generation. Name: {self.name}")
        if deep > self._order_max_deep:
            raise MaximumOrderDeepError(self.name)
        if checked is None:
            checked = {}

        res = True

        if not self.is_one():

            # check whether any of cyclic shifts of name was checked
            # elements with same names up to cycle shifts are conjugate
            # and therefore have same order
            found = self._check_cycle_shifts(self.name, checked)
            prev_power = checked.get(found)
            if found and prev_power != cur_power:
                if verbose:
                    print(f'Found cycle between {self.name} and {found} '
                          f'of length {cur_power / prev_power}')
                return False
            elif found:
                return True
            checked[self.name] = cur_power

            orbits = self._get_orbits(self.permutation)
            # always start from orbit of 0
            for orbit in sorted(orbits):
                if check_only is not None and check_only not in orbit:
                    continue

                power = len(orbit)
                next_el = (self ** power)[orbit[0]]
                if not next_el._is_finite_dfs(cur_power=cur_power * power,
                                              checked=checked,
                                              check_only=check_only,
                                              deep=deep+1, verbose=verbose):
                    res = False
                    break
            del checked[self.name]

        return res

    def _is_finite_bfs(self, check_only=False,
                       verbose=False):
        queue = deque()
        queue.append((self, {}, 1, 1))
        while queue:
            el, checked, cur_power, deep \
                = queue.popleft()  # type: AutomataGroupElement, dict, int, int

            if verbose:
                print(f'Generation: {deep}, name: {el.name}')
            if el.is_one():
                continue

            if deep > self._order_max_deep:
                raise MaximumOrderDeepError(self.name)

            # check whether any of cyclic shifts of name was checked
            # elements with same names up to cycle shifts are conjugate
            # and therefore have same order
            found = self._check_cycle_shifts(el.name, checked)

            prev_power, prev_deep = checked.get(found, (1, 1))
            # if we found cycle of non-unitary length it means that
            # all of predecessors are infinite elements
            if found and prev_power != cur_power:
                if verbose:
                    print(f'Found cycle between {el.name} and {found} '
                          f'of length {cur_power / prev_power}')

                self._cycle_start_el = found
                self._cycle_end_el = el.name
                self._cycle_start_deep = prev_deep
                self._cycle_end_deep = deep
                self._cycle_start_power = prev_power
                self._cycle_end_power = cur_power
                self._cycle_len = cur_power / prev_power

                res = False
                for prev in checked:
                    prev = self.parent_group(prev)
                    if prev._is_finite is None:
                        prev._is_finite = False
                    else:
                        assert not prev._is_finite, f'Better check {prev}'
                return res

            elif found:
                if verbose:
                    print(f'Found cycle on {el.name} '
                          f'of length {cur_power / prev_power}')
                continue

            checked[el.name] = cur_power, deep

            orbits = self._get_orbits(el.permutation)
            for orbit in sorted(orbits):
                if check_only is not None and check_only not in orbit:
                    continue

                power = len(orbit)
                next_el = (el ** power)[orbit[0]]
                queue.append((next_el, checked.copy(), cur_power * power, deep + 1))
        return True

    @staticmethod
    def _get_orbits(permutation):
        fixed_points = [[i] for i in range(permutation.size)
                        if permutation(i) == i]
        cycles = permutation.cyclic_form
        return fixed_points + cycles

    @staticmethod
    def _check_cycle_shifts(el_name: str, checked):
        for i in range(len(el_name)):
            tmp_name = el_name[i:] + el_name[:i]
            if tmp_name in checked:
                return tmp_name
        return ''

    @_Decorators.check_group
    def order_graph(self, graph, loops=False, as_tree=False, max_deep=10, short_names=True):
        added = defaultdict(int)
        checked = {}
        added[self.name] += 1
        cur_vertex = self._create_name(self.name, added[self.name], short_names)
        graph.add_vertex(cur_vertex)
        checked[self.name] = cur_vertex
        self._order_graph(graph=graph, checked=checked, added=added,
                          loops=loops, as_tree=as_tree, deep=1,
                          max_deep=max_deep, short_names=short_names)

    def _order_graph(self, graph, checked, added, loops, as_tree, deep=1,
                     max_deep=10, short_names=True):
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

            found = self._check_cycle_shifts(next_el.name, checked)

            if not loops and found == self.name:
                continue

            if found:
                if as_tree:
                    continue
                dest = checked[found]
                graph.add_edge(cur_vertex, dest, power)
            else:
                added[next_el.name] += 1
                next_vertex = self._create_name(next_el.name,
                                                added[next_el.name], short_names)
                graph.add_vertex(next_vertex)
                checked[next_el.name] = next_vertex
                graph.add_edge(cur_vertex, next_vertex, power)
                next_el._order_graph(graph=graph, checked=checked,
                                     added=added, loops=loops, as_tree=as_tree,
                                     deep=deep+1, max_deep=max_deep,
                                     short_names=short_names)
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

    # TODO: add unit tests
    def describe(self, graph_class=None, show_structure=True,
                 y_scale_mul=5, max_deep=7, loops=True,
                 figsize=(15, 15), vertex_size=15):
        self.disable_cache()
        tmp = f"""
        {str(self)}
        Group:     {self.parent_group}
        size:      {self.tree.size()}
        height:    {self.tree.height()}
        
        is finite: {self.is_finite()}
        order:     {self.order()} 
        """
        tmp2 = f"""
        Found cycle 
            start deep:   {self._cycle_start_deep}
            end deep:     {self._cycle_end_deep}
            start el:     {self._cycle_start_el}
            end el:       {self._cycle_end_el}
            start power:  {self._cycle_start_power}
            end power:    {self._cycle_end_power}
            cycle weight: {self._cycle_len}
        """
        self.enable_cache()
        if self._cycle_start_el is not None:
            tmp += tmp2
        print(tmp)
        if show_structure:
            self.show(y_scale_mul=y_scale_mul)
        if graph_class is not None:
            graph = graph_class(loops=loops, multiedges=True)
            self.order_graph(graph, max_deep=max_deep, loops=loops)
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
                reduce_function=id_func):
        if not gens:
            raise ValueError("Gens should be a non-empty list of AutomataGroupElement-s")

        if name not in cls.__instances:
            obj = super(AutomataGroup, cls).__new__(cls)
            obj.name = name
            obj._defined_els = TriedDict()
            obj._defined_trees = {}

            obj.__gens = gens
            obj._size = gens[-1].permutation.size
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
                left_word = word
                last_value = self._e
                while left_word:
                    prefix, left_word, value = self._defined_els.max_prefix(left_word)
                    if prefix not in self._defined_els:
                        raise ValueError(f"Unknown prefix: {prefix}")
                    res *= value
                    _ = last_value * value     # cache last two prefixes
                    last_value = value
                self._defined_els[word] = res
            return self._defined_els[word]
        elif isinstance(word, AutomataGroupElement):
            return self.__call__(word.name)
        else:
            raise TypeError(f"Not supported type: {type(word)}")

    def multiply(self, x1, x2):

        if x1.name == 'e':
            return x2
        if x2.name == 'e':
            return self

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

    def __str__(self):
        return self.name

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
