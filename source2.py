#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 04.01.2020
# by David Zashkolny

# Excusa. Quod scripsi, scripsi.
# 3 course, comp math
# Taras Shevchenko National University of Kyiv
# email: davendiy@gmail.com

from trees import *
from sympy.combinatorics import Permutation
from collections import deque
import numpy as np


TRIVIAL_PERM = Permutation([0, 1, 2])


# TODO: YOU MUST ADD COMMENTS


class AutomataTreeNode(Tree):

    _colors = {        # color of the vertex depends on its permutation
        '(2)': 'y',    # could be changed into 'unique color for each permutation'
        '(0 1 2)': 'r',  # now color is unique for each possible order of permutation
        '(0 2 1)': 'r',
        '(2)(0 1)': 'b',
        '(0 2)': 'b',
        '(1 2)': 'b'
    }

    _labels = {           # corresponding labels for colors
        'r': 'permutation of order 3',
        'b': 'permutation of order 2',
        'y': 'trivial permutation',
        'k': 'reverse node'
    }

    def __init__(self, permutation=TRIVIAL_PERM, value=None, reverse=False,
                 simplify=False):
        self.permutation = permutation
        self.reverse = reverse
        self.simplify = simplify
        if reverse:
            value = 'δ'
        elif value is None:
            value = str(permutation)
        super().__init__(value)

    @property
    def name(self):
        return self.value

    def get_coords(self, start_x, start_y, scale, deep=0, show_full=False,
                   y_scale_mul=3):
        x_coord = start_x + self._offset * scale
        y_coord = start_y - deep * scale * y_scale_mul
        yield x_coord, y_coord

        if not self.simplify or show_full:
            for child in self.children:  # type: AutomataTreeNode
                for coords in child.get_coords(start_x, start_y, scale,
                                               deep + 1, show_full=show_full,
                                               y_scale_mul=y_scale_mul):
                    yield coords
                yield x_coord, y_coord

    def draw(self, start_x=0, start_y=0, scale=10, radius=4, fontsize=50,
             save_filename='', show_full=False, y_scale_mul=3, lbn=False):
        fig, ax = plt.subplots(figsize=(20, 20))
        # ax = fig.add_subplot(111)

        self.make_offsets()

        x_coords = []
        y_coords = []

        for x, y in self.get_coords(start_x, start_y, scale, show_full=show_full,
                                    y_scale_mul=y_scale_mul):
            x_coords.append(x)
            y_coords.append(y)

        fontsize = fontsize / (np.log(self.vert_amount()))

        # draw edges
        ax.plot(x_coords, y_coords, linewidth=0.5)

        used_colors = set()
        # draw vertices
        self._draw(ax, start_x, start_y, scale, radius, fontsize, show_full=show_full,
                   y_scale_mul=y_scale_mul, used_colors=used_colors, lbn=lbn)

        for color in used_colors:
            circle = plt.Circle((0, 0), radius=radius/10,
                                color=color, label=self._labels[color])
            ax.add_patch(circle)
        circle = plt.Circle((0, 0), radius=radius/10, color='w')
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
              lbn=False):

        if used_colors is None:
            used_colors = set()

        x_coord = self._offset * scale + start_x
        y_coord = start_y - deep * scale * y_scale_mul

        if not self.reverse:
            color = AutomataTreeNode._colors[str(self.permutation)]
            circle = plt.Circle((x_coord, y_coord), radius=radius, color=color)
            used_colors.add(color)
        else:
            circle = plt.Circle((x_coord, y_coord), radius=radius, color='k')
            used_colors.add('k')
        ax.add_patch(circle)

        if (not self.children or (self.simplify and not show_full)) and lbn:
            bias = -2
        else:
            bias = 1
        ax.annotate(str(self.value), xy=(x_coord, y_coord + radius * bias),
                    xytext=(-fontsize * (len(str(self.value))//3), 2 * bias),
                    textcoords='offset pixels',
                    fontsize=fontsize)

        if not self.simplify or show_full:
            for child in self.children:       # type: AutomataTreeNode
                child._draw(ax, start_x, start_y, scale, radius, fontsize,
                            deep + 1, show_full, y_scale_mul, used_colors, lbn)

    def add_child(self, child, position=None):
        if isinstance(child, AutomataTreeNode):
            _child = child.copy()       # add only copies of the given trees
        else:                           # in order to avoid cycles
            raise NotImplementedError()
        if position is None:
            self.children.append(_child)
        else:
            self.children.insert(position, _child)
        _child.parent = self
        return _child

    def remove(self, child):
        if isinstance(child, AutomataTreeNode):
            child.parent = None
            self.children.remove(child)
        else:
            for i, el in enumerate(self.children):
                if el.value == child:
                    el.parent = None
                    del self.children[i]
                    break

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        res = AutomataTreeNode(self.permutation, self.value,
                               self.reverse, self.simplify)
        for child in self.children:   # type: AutomataTreeNode
            res.add_child(child)
        return res


class AutomataGroupElement:

    __instances = {}

    def __new__(cls, name, *args, **kwargs):
        if name not in AutomataGroupElement.__instances:
            AutomataGroupElement.__instances[name] = \
                super(AutomataGroupElement, cls).__new__(cls)
        return AutomataGroupElement.__instances[name]

    def __init__(self, name, permutation=TRIVIAL_PERM,
                 children=(AutomataTreeNode(reverse=True),
                           AutomataTreeNode(reverse=True),
                           AutomataTreeNode(reverse=True)),
                 simplify=False):
        self.name = name

        self.tree = AutomataTreeNode(permutation=permutation,
                                     value=name, simplify=simplify)
        for child in children:
            self.add_child(child)

    @classmethod
    def from_cache(cls, el):
        if el not in cls.__instances:
            raise ValueError(f"Not initialized element: {el}")
        return cls.__instances[el]

    def add_child(self, child):
        if isinstance(child, AutomataTreeNode):
            local_child = child.copy()
        elif isinstance(child, AutomataGroupElement):
            local_child = child.tree.copy()
        else:
            raise TypeError(f"Bad type for child: {type(child)}.")
        self.tree.add_child(local_child)

    @property
    def permutation(self) -> Permutation:
        return self.tree.permutation

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise KeyError(f"Bad key: {item}.")
        res_tree = self.tree[item]    # type: AutomataTreeNode
        if res_tree.reverse:
            return self
        else:
            return AutomataGroupElement(res_tree.name, res_tree.permutation,
                                        res_tree.children, res_tree.simplify)

    def is_trivial(self):
        if self.name == 'e':
            return True
        if self.permutation != TRIVIAL_PERM:
            return False

        queue = deque()
        queue.extend(self.tree.children)
        res = True
        while queue:
            child = queue.pop()   # type: AutomataTreeNode
            if child.reverse:
                continue
            if child.permutation != TRIVIAL_PERM:
                res = False
                break
            queue.extend(child.children)
        return res

    def __repr__(self):

        children = []
        for child in self.tree.children:
            if child.reverse:
                children.append(self.name)
            else:
                children.append(child.name)
        return f'{self.name} = {self.permutation} ({", ".join(children)})'

    def __str__(self):
        return self.__repr__()

    def __mul__(self, other):
        if not isinstance(other, AutomataGroupElement):
            raise ValueError("Bad type for multiplier.")

        if self.name == 'e':
            return other
        if other.name == 'e':
            return self

        res_name = self.name + other.name
        if res_name in self.__instances:
            return self.__instances[res_name]

        res_permutation = other.permutation * self.permutation
        res_children = []
        for i in range(len(self.tree.children)):
            self_child = self[i ^ other.permutation]  # type: AutomataGroupElement
            other_child = other[i]                    # type: AutomataGroupElement
            if self_child.name + other_child.name == res_name:
                res_children.append(AutomataTreeNode(reverse=True))
            else:
                res_children.append(self_child * other_child)
        return AutomataGroupElement(res_name, res_permutation, res_children)

    def show(self, start_x=0, start_y=0, scale=10, radius=4, fontsize=50,
             save_filename='', show_full=False, y_scale_mul=3, lbn=False):
        self.tree.draw(start_x=start_x, start_y=start_y,
                       scale=scale, radius=radius,
                       fontsize=fontsize, y_scale_mul=y_scale_mul,
                       save_filename=save_filename, show_full=show_full, lbn=lbn)


def from_string(string) -> AutomataGroupElement:
    res = AutomataGroupElement.from_cache('e')
    for el in string:
        res *= AutomataGroupElement.from_cache(el)
    return res


def reverse_node():
    return AutomataTreeNode(reverse=True)


e = AutomataGroupElement('e', simplify=True)
a = AutomataGroupElement('a', permutation=Permutation([1, 0, 2]),
                         children=(e, e, reverse_node()), simplify=True)
b = AutomataGroupElement('b', permutation=Permutation([2, 1, 0]),
                         children=(e, reverse_node(), e), simplify=True)
c = AutomataGroupElement('c', permutation=Permutation([0, 2, 1]),
                         children=(reverse_node(), e, e), simplify=True)


if __name__ == '__main__':
    e.show(show_full=True)
    a.show(show_full=True)
    b.show(show_full=True)
    c.show(show_full=True)

    ab = a * b
    abc = ab * c
    abc.show(show_full=True)
    abc.show()
    abc.show(show_full=True)

    abcc = abc * c
    abcc.show()

    test = from_string('cbcbacabacacbcabacabac')
    test.show(save_filename='graphs/test.png', show_full=True)
