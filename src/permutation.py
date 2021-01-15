#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 15.01.2021
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

from functools import reduce
from .tools import lcm


class Permutation(tuple):

    __instances = {}

    def __new__(cls, form):

        if isinstance(form, list):
            array_form = tuple(form)
        elif isinstance(form, int):
            array_form = tuple(range(form))
        elif isinstance(form, list):
            array_form = form
        else:
            raise TypeError(f"Permutation should be list, tuple or int, not {type(form)}")

        if array_form not in cls.__instances:
            obj = tuple.__new__(cls, array_form)
            obj._cyclic_form = None
            obj._order = None
            cls.__instances[array_form] = obj
        return cls.__instances[array_form]

    @property
    def size(self):
        return len(self.array_form)

    @property
    def array_form(self):
        return list(self)

    def __call__(self, el):
        if not 0 <= el < self.size:
            raise ValueError(f"Called value is out of bounds.")
        return self.array_form[el]

    def __mul__(self, other):
        if not isinstance(other, Permutation):
            raise TypeError(f"Multiplier should be instance of Permutation class, not {type(other)}.")

        a = self.array_form
        b = other.array_form
        if not b:
            perm = a
        else:
            b.extend(list(range(len(b), len(a))))
            perm = [b[i] for i in a] + b[len(a):]
        return Permutation(perm)

    @property
    def cyclic_form(self):
        if self._cyclic_form is not None:
            return self._cyclic_form
        array_form = self.array_form
        unchecked = [True] * len(array_form)
        cyclic_form = []
        for i in range(len(array_form)):
            if unchecked[i]:
                cycle = [i]
                unchecked[i] = False
                j = i
                while unchecked[array_form[j]]:
                    j = array_form[j]
                    cycle.append(j)
                    unchecked[j] = False
                if len(cycle) > 1:
                    cyclic_form.append(cycle)
        cyclic_form.sort()
        self._cyclic_form = cyclic_form[:]
        return cyclic_form

    def order(self):
        if self._order is None:
            self._order = reduce(lcm, [len(cycle) for cycle in self.cyclic_form], 1)
        return self._order

    def __str__(self):
        if self.size == 0:
            return '()'
        cycles = self.cyclic_form
        big = self.size - 1
        s = ''
        if not any(i == big for c in cycles for i in c):
            s += '(%s)' % big
        s += ''.join(str(tuple(c)) for c in cycles)
        s = s.replace(',', '')
        return s
