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
    """ Recursive tree structure, capable to be drawn in matplotlib.

    Object of this class represents a tree node with value and any amount
    of children.

    Parameters
    ----------
    value     : str, value that will be shown on the plot
    children  : iterable of Tree objects

    Examples
    --------

    Notes
    -----

    TODO: add examples and notes
    """

    def __init__(self, value, children=()):
        self.value = value
        self.children = deque()      # list of children

        # vertex horizontal offset relative to zero (the most left point)
        # it's just for plotting
        self._offset = 0

        for child in children:
            self.add_child(child)

    def copy(self):
        """ Create copy of the entire tree recursively.
        """
        res = Tree(self.value)
        for child in self.children:  # type: Tree
            res.add_child(child)
        return res

    def add_child(self, child, position=None):
        """ Add copy of the child to the tree.

        Parameters
        ----------
        child: Tree or any another type
               If one has type of Tree, it will be copied and
               added (inserted) to the deque of children
               Else new Tree with the given value will be created
        position: index where we must insert the given child
                  (default None means append to the end)

        Returns
        -------
        Created or copied subtree
        """
        if isinstance(child, Tree):
            _child = child.copy()       # add only copies of the given trees
        else:                           # in order to avoid cycles
            _child = Tree(child)  # if the type is different - create new Tree
        if position is None:
            self.children.append(_child)
        else:
            self.children.insert(position, _child)
        return _child

    def height(self):
        """ Just height of tree, calculated using dfs.

        Notes
        -----
        Returned value isn't cached, therefore dfs will be run anyway.
        """
        res = 0
        for child in self.children:  # type: Tree
            res = max(child.height(), res)
        return res + 1

    def vert_amount(self) -> int:
        """ Calculates the amount of all the vertices recursively
        using dfs.

        Notes
        -----
        Returned value isn't cached, therefore dfs will be run anyway.
        """
        res = 1
        for el in self.children:    # type: Tree
            res += el.vert_amount()
        return res

    def remove(self, child):
        """ Removes child from tree.

        Parameters
        ----------
        child : Tree or value of tree

        Notes
        -----
        If there are more than one children with same value, only the first one
        will be deleted if parameter child represents value.
        """
        if isinstance(child, Tree):
            self.children.remove(child)
        else:
            for i, el in enumerate(self.children):
                if el.value == child:
                    del self.children[i]
                    break

    def draw(self, start_x=0, start_y=0, scale=10, radius=2, fontsize=10,
             save_filename=''):
        """ Draws the tree in matplotlib.

        Parameters
        ----------
        start_x  : x coordinate of the start position on the plane
        start_y  : y coordinate of the start position on the plane
        scale    : length of one step of offset.
                   Offset is measure of vertices' displacement relative to
                   left upper corner. Distance between 2 generation == one
                   step of offset.
        radius   : radius of vertices
        fontsize : size of font (like fontsize in matplotlib.pyplot.text)
        save_filename : name of file, where it should be save.
                        Default == '', means that the picture won't be saved

        Returns
        -------
        None

        TODO: add examples and notes
        Examples
        --------

        Notes
        -----
        """
        fig, ax = plt.subplots()
        # ax = fig.add_subplot(111)

        self.make_offsets()       # pre-calculate the offsets
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
            fig.savefig(save_filename)

    def _draw(self, ax, start_x, start_y, scale, radius,
              fontsize, deep=0):
        """ Auxiliary recursive (DFS) function for drawing the vertices of tree.

        Parameters
        ----------
        start_x  : x coordinate of the start position on the plane
        start_y  : y coordinate of the start position on the plane
        scale    : length of one step of offset.
                   Offset is measure of vertices' displacement relative to
                   left upper corner. Distance between 2 generation == one
                   step of offset.
        radius   : radius of vertices
        fontsize : size of font (like fontsize in matplotlib.pyplot.text)
        deep : auxiliary parameter - number of self's generation

        Returns
        -------
        None
        """
        x_coord = self._offset * scale + start_x
        y_coord = start_y - deep * scale

        circle = plt.Circle((x_coord, y_coord), radius=radius, color='r')
        ax.add_patch(circle)
        ax.annotate(str(self.value), xy=(x_coord, y_coord),
                    fontsize=fontsize)

        for child in self.children:       # type: Tree
            child._draw(ax, start_x, start_y, scale, radius, fontsize, deep + 1)

    def get_coords(self, start_x, start_y, scale, deep=0):
        """ Recursively (DFS) generates sequence of coordinates of vertices for
        drawing the edges.

        Parameters
        ----------
        start_x : x coordinate of the start position on the plane
        start_y : y coordinate of the start position on the plane
        scale   : length of one step of offset.
                  Offset is measure of vertices' displacement relative to
                  left upper corner. Distance between 2 generation == one
                  step of offset.
        deep    : auxiliary parameter - number of self's generation

        Yields
        ------
        (x, y) - coordinates on the plane
        """
        x_coord = start_x + self._offset * scale
        y_coord = start_y - deep * scale
        yield x_coord, y_coord
        for child in self.children:  # type: Tree
            for coords in child.get_coords(start_x, start_y, scale, deep + 1):
                yield coords
            yield x_coord, y_coord

    def make_offsets(self):
        """ Calculates offsets of all the vertices relative to zero point -
        the left bound. Uses for drawing without overlaps.
        """
        pre_shifted = 0
        pre_right_bound = 0
        for child in self.children:  # type: Tree
            child.make_offsets()

            # left and right bounds of child ('some kind of width')
            next_left_bound, next_right_bound = child._max_offsets()

            # shifting of i-th child to right on shift
            # amount of steps equals to amount of vertices that will
            # overlap with previous child
            shift = (pre_right_bound + 1) - next_left_bound + pre_shifted
            child._shift(shift)

            # update values
            pre_right_bound = next_right_bound
            pre_shifted = shift

        # if child only one, we just place the parent on one step above
        if len(self.children) == 1:
            self._offset = self.children[0]._offset

        # else we must place the parent at the middle of line between left and
        # right children
        elif len(self.children) > 1:
            self._offset = self.children[0]._offset + self.children[-1]._offset
            self._offset /= 2

    def _max_offsets(self) -> tuple:
        """ Calculates the most left and the most right biases of
        tree (in ones of offset).
        """
        _min_left = _max_right = self._offset
        for child in self.children:  # type: Tree
            next_left, next_right = child._max_offsets()
            _min_left = min(_min_left, next_left)
            _max_right = max(_max_right, next_right)
        return _min_left, _max_right

    def _shift(self, points):
        """ Shifts the tree on the 'points' steps right.
        """
        self._offset += points
        for child in self.children:  # type: Tree
            child._shift(points)

    def __getitem__(self, item):
        return self.children[item]

    def __setitem__(self, key, value):
        if not isinstance(value, Tree):
            value = Tree(value)

        self.children[key] = value

    def __delitem__(self, key):
        child = self.children[key]  # type: Tree
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
    test_tmp = test_tree.children[-1].add_child('x')
    test_tmp.add_child('x')
    test_tmp.add_child('x')

    test_tree.add_child('y')

    test_tree.add_child(test_tree)
    test_tree.add_child(test_tree, position=0)
    test_tree.draw(save_filename='../graphs/test.png')
