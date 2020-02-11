#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 18.01.2020
# Excusa. Quod scripsi, scripsi.

# by David Zashkolny
# 3 course, comp math
# Taras Shevchenko National University of Kyiv
# email: davendiy@gmail.com

from collections import deque
import matplotlib.pyplot as plt


class Tree:

    def __init__(self, value, children=()):
        self.value = value
        self.children = deque()      # list of children
        self.parent = None

        self._offset = 0  # vertex offset relative to zero (zero should be root)
        self._max_right = None  # offset of the most right descendant (not child)
        self._min_left = None   # offset of the most left descendant (not child)

        for child in children:
            self.add_child(child)

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        res = Tree(self.value)
        for child in self.children:   # type: Tree
            res.add_child(child)
        return res

    def add_child(self, child, position=None):
        if isinstance(child, Tree):
            _child = child.copy()       # add only copies of the given trees
        else:                           # in order to avoid cycles
            _child = Tree(child)
        if position is None:
            self.children.append(_child)
        else:
            self.children.insert(position, _child)
        _child.parent = self
        return _child

    def height(self, start=0):
        res = start
        for el in self.children:    # type: Tree
            res = max(res, el.height(start=start+1))
        return res

    def vert_amount(self):
        res = 1
        for el in self.children:    # type: Tree
            res += el.vert_amount()
        return res

    def remove(self, child):
        if isinstance(child, Tree):
            child.parent = None
            self.children.remove(child)
        else:
            for i, el in enumerate(self.children):
                if el.value == child:
                    el.parent = None
                    del self.children[i]
                    break

    def draw(self, start_x=0, start_y=0, scale=10, radius=2, fontsize=10,
             save_filename=''):
        fig, ax = plt.subplots()
        ax = fig.add_subplot(111)

        self.make_offsets()

        x_coords = []
        y_coords = []
        for x, y in self.get_coords(start_x, start_y, scale):
            x_coords.append(x)
            y_coords.append(y)

        # draw edges
        ax.plot(x_coords, y_coords, linewidth=0.2)

        # draw vertices
        self._draw(ax, start_x, start_y, scale, radius, fontsize)

        ax.axis('off')
        ax.set_aspect('equal')
        ax.autoscale_view()

        fig.show()
        if save_filename:
            fig.savefig('test.png')

    def _draw(self, ax, start_x, start_y, scale, radius,
              fontsize, deep=0):

        x_coord = self._offset * scale + start_x
        y_coord = start_y - deep * scale

        circle = plt.Circle((x_coord, y_coord), radius=radius, color='r')
        ax.add_patch(circle)
        ax.annotate(str(self.value), xy=(x_coord, y_coord),
                    fontsize=fontsize)

        for child in self.children:       # type: Tree
            child._draw(ax, start_x, start_y, scale, radius, fontsize, deep + 1)

    def get_coords(self, start_x, start_y, scale, deep=0):
        x_coord = start_x + self._offset * scale
        y_coord = start_y - deep * scale
        yield x_coord, y_coord
        for child in self.children:  # type: Tree
            for coords in child.get_coords(start_x, start_y, scale, deep + 1):
                yield coords
            yield x_coord, y_coord

    def make_offsets(self):
        pre_shifted = 0
        pre_right_bound = 0
        for child in self.children:  # type: Tree
            child.make_offsets()

            next_left_bound, next_right_bound = child._max_offsets()
            shift = (pre_right_bound + 1) - next_left_bound + pre_shifted

            # print(f'shifted {child.value} on {shift}')
            child._shift(shift)

            pre_right_bound = next_right_bound
            pre_shifted = shift
        if len(self.children) == 1:
            self._offset = self.children[0]._offset
        elif len(self.children) > 1:
            self._offset = self.children[0]._offset + self.children[-1]._offset
            self._offset /= 2

    def _max_offsets(self) -> tuple:
        _min_left = _max_right = self._offset
        for child in self.children:  # type: Tree
            next_left, next_right = child._max_offsets()
            _min_left = min(_min_left, next_left)
            _max_right = max(_max_right, next_right)
        return _min_left, _max_right

    def _shift(self, points):
        self._offset += points
        for child in self.children:  # type: Tree
            child._shift(points)

    def __getitem__(self, item):
        return self.children[item]

    def __setitem__(self, key, value):
        if not isinstance(value, Tree):
            value = Tree(value)

        self.children[key] = value
        value.parent = self

    def __delitem__(self, key):
        child = self.children[key]  # type: Tree
        child.parent = None
        del self.children[key]


if __name__ == '__main__':
    test_tree = Tree('(1, 2)')
    test_tree.add_child('b')
    test_tree.add_child('c')
    test_tree.add_child('c')
    for tmp_child in test_tree.children:  # type: Tree
        tmp_child.add_child('e')
        tmp_child.add_child('e')
        tmp_child.add_child('e')

    test_tree.add_child('d')
    test_tree.children[-1].add_child('x')
    test_tree.children[-1].add_child('x')
    tmp = test_tree.children[-1].add_child('x')
    tmp.add_child('x')
    tmp.add_child('x')

    test_tree.add_child('y')

    test_tree.add_child(test_tree)
    test_tree.add_child(test_tree, position=0)
    test_tree.draw(save_filename='graphs/test.png')
