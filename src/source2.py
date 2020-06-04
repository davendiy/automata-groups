#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 21.02.2020
# Excusa. Quod scripsi, scripsi.

# by David Zashkolny
# email: davendiy@gmail.com


from src.trees import *
from sympy.combinatorics import Permutation
from collections import deque
import numpy as np
import psutil
import os

# TODO: replace all the refs in trees with weakrefs, or
#  implement the delete method in right way


TRIVIAL_PERM = Permutation([0, 1, 2])
CHUNK = 1 * 1024 * 1024 * 1024

process = psutil.Process(os.getpid())


# TODO: YOU MUST ADD COMMENTS


class AutomataTreeNode(Tree):

    # color of the vertex depends on its permutation
    # could be changed into 'unique color for each permutation'
    # now color is unique for each possible order of permutation
    _colors = {
        '(2)': 'y',
        '(0 1 2)': 'r',
        '(0 2 1)': 'r',
        '(2)(0 1)': 'b',
        '(0 2)': 'b',
        '(1 2)': 'b'
    }

    # corresponding labels for colors
    _labels = {
        'r': 'permutation of order 3',
        'b': 'permutation of order 2',
        'y': 'trivial permutation',
        'k': 'reverse node'
    }

    def __init__(self, permutation=TRIVIAL_PERM, value=None, reverse=False,
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
        self._height = None
        self._vert_amount = None
        self._size = None

        if reverse:
            value = 'δ'      # delta is just similar to spiral arrow (like reverse in uno)
        elif value is None:
            value = str(permutation)
        super().__init__(value)

    @property
    def name(self):
        return self.value

    def height(self) -> int:
        if self._height is None:
            self._height = 0
            for child in self.children:  # type: AutomataTreeNode
                self._height = max(child.height(), self._height)
            self._height += 1
        return self._height

    def vert_amount(self) -> int:
        """ Calculates the amount of all the vertices recursively.
        """
        if self._vert_amount is None:
            self._vert_amount = 1
            for el in self.children:    # type: AutomataTreeNode
                self._vert_amount += el.vert_amount()
        return self._vert_amount

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
        # size when matplotlib will use autoscale for picture
        fontsize = fontsize / (np.log(self.vert_amount()))
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
            color = AutomataTreeNode._colors[str(self.permutation)]
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
        res._size = self._size
        res._height = self._height
        res._vert_amount = self._vert_amount
        for child in self.children:   # type: AutomataTreeNode
            res.add_child(child)
        return res


class AutomataGroupElement:
    """ Implementation of an element of an automatic group.

    This class represents any element that has structure
    g = \\pi (g_1, g_2, g_3). Group defines by initialising an alphabet of
    its generators {a1, a2, a3 ...} using __init__ method and then for every
    word w from the alphabet corresponding element can be created using
    function $$$ from_string $$$.

    a = AutomataGroupElement(name, permutation, children)

    :param name: the name of an atom (for instance a, b, c, e for H3)
    :param permutation: object sympy.combinatorics.Permutation
    :param children: a list of AutomataTreeNode or AutomataGroupElement
                     elements that defines tree-like structure of the element
                     (first-level state). Those elements should have $ name $
                     of the respective atom or have parameter $ reverse $ with
                     value True that means recursive call.

    :param simplify: True if it could be drawn as point with name
                         (e.g. for e, a, b, c)

    Example of using:
    >>> e = AutomataGroupElement('e', simplify=True)
    >>> a = AutomataGroupElement('a', permutation=Permutation([1, 0, 2]), \
                                 children=(e, e, reverse_node()), simplify=True)
    >>> b = AutomataGroupElement('b', permutation=Permutation([2, 1, 0]), \
                                 children=(e, reverse_node(), e), simplify=True)
    >>> c = AutomataGroupElement('c', permutation=Permutation([0, 2, 1]), \
                                 children=(reverse_node(), e, e), simplify=True)
    >>> from_string('abcbabcbabcbabc')
    abcbabcbabcbabc = (0 2) (acacacac, bbb, bbbb)


    As you can see element is completely defined by it's tree structure.
    The multiplication of two elements is just combining of their trees in some way.

    WARNING: you can't create in such way elements that reffers to each other,
    for example
                 a = ()(a, b, e)      and  b = ()(b, e, a)
    because such elements don't have tree-like structure.
    Well in fact, you can do it but I don't guarantee it will work properly.
    """

    __instances = {}

    @classmethod
    def defined_size(cls):
        return len(cls.__instances)

    @classmethod
    def clear_memory(cls):
        del cls.__instances
        cls.__instances = {}

    def __new__(cls, name, *args, **kwargs):
        if name not in AutomataGroupElement.__instances and \
                                    process.memory_info()[0] < CHUNK:
            AutomataGroupElement.__instances[name] = \
                super(AutomataGroupElement, cls).__new__(cls)
            return AutomataGroupElement.__instances[name]
        else:
            return super(AutomataGroupElement, cls).__new__(cls)

    def __init__(self, name, permutation=TRIVIAL_PERM,
                 children=(AutomataTreeNode(reverse=True),
                           AutomataTreeNode(reverse=True),
                           AutomataTreeNode(reverse=True)),
                 simplify=False):
        """ Initialisation of atomic elements.

        :param name: the name of an atom (for instance a, b, c, e for H3)
        :param permutation: object sympy.combinatorics.Permutation
        :param children: a list of AutomataTreeNode elements that defines
                         tree-like structure of the element. Those elements
                         should have $ name $ of the respective
                         atom or have parameter $ reverse $ with value True
                         that means recursive call.
        :param simplify: True if it could be drawn as point with name
                         (e.g. for e, a, b, c)
        """

        self.name = name

        self.tree = AutomataTreeNode(permutation=permutation,
                                     value=name, simplify=simplify)
        for child in children:
            self.add_child(child)

    def __len__(self):
        if self.name == 'e':
            return 0
        else:
            return len(self.name)

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

    def __iter__(self):
        for subtree in self.tree.children:
            if subtree.reverse:
                yield self
            else:
                yield AutomataGroupElement(subtree.name, subtree.permutation,
                                           subtree.children, subtree.simplify)

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

        res_permutation = other.permutation * self.permutation    # other(self())
        res_children = []
        for i in range(len(self.tree.children)):
            self_child = self[i ^ other.permutation]  # type: AutomataGroupElement
            other_child = other[i]                    # type: AutomataGroupElement
            if self_child.name + other_child.name == res_name:
                res_children.append(AutomataTreeNode(reverse=True))
            else:
                res_children.append(self_child * other_child)
        return AutomataGroupElement(res_name, res_permutation, res_children)

    def __pow__(self, power):
        if power == -1:
            return from_string(self.name[::-1])
        else:
            res = AutomataGroupElement.from_cache('e')
            tmp = self
            i = 1
            while i <= power:
                if i & power:
                    res *= tmp
                i <<= 1
                tmp *= tmp
            return res

    def order(self):
        if self.is_trivial():
            return 1

        perm_order = self.tree.permutation.order()
        tmp = self ** int(perm_order)
        if (tmp[0].name == tmp[0].name[::-1]
                and tmp[1].name == tmp[1].name[::-1]
                and tmp[2].name == tmp[2].name[::-1]):
            return perm_order * 2 if not tmp.is_trivial() else perm_order
        else:
            return float('inf')

    def show(self, start_x=0, start_y=0, scale=10, radius=4, fontsize=50,
             save_filename='', show_full=False, y_scale_mul=3, lbn=False,
             show_names=True):
        self.tree.draw(start_x=start_x, start_y=start_y,
                       scale=scale, radius=radius,
                       fontsize=fontsize, y_scale_mul=y_scale_mul,
                       save_filename=save_filename, show_full=show_full, lbn=lbn,
                       show_names=show_names)


def from_string(w: str) -> AutomataGroupElement:
    """ Get element that related to the given string.

    For example, you can define atomic elements 'a' 'b' and 'c' and then
    get an element 'ababababacbcbabc'

    Complexity - O(n * V), where n = |w| and V = v(w) (amount of vertices)

    :param w: a word over an alphabet formed by atomic elements.
    """
    res = AutomataGroupElement.from_cache('e')
    for el in w:
        res *= AutomataGroupElement.from_cache(el)
    return res


def reverse_node():
    return AutomataTreeNode(reverse=True)


def initial_state():
    global e, a, b, c
    e = AutomataGroupElement('e', simplify=True)
    a = AutomataGroupElement('a', permutation=Permutation([1, 0, 2]),
                             children=(e, e, reverse_node()), simplify=True)
    b = AutomataGroupElement('b', permutation=Permutation([2, 1, 0]),
                             children=(e, reverse_node(), e), simplify=True)
    c = AutomataGroupElement('c', permutation=Permutation([0, 2, 1]),
                             children=(reverse_node(), e, e), simplify=True)


def permute(seq, repeat):
    if repeat == 1:
        for el in seq:
            yield [el]
    elif repeat < 1:
        yield []
    else:
        for prev in permute(seq, repeat-1):
            for el in seq:
                if prev[-1] == el:
                    continue
                yield prev + [el]


e = a = b = c = ...     # type: AutomataGroupElement
initial_state()

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
    test.show(save_filename='../graphs/test.png', show_full=True)
