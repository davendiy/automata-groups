#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 23.03.2021
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

from functools import reduce
from .tools import lcm


class Permutation(tuple):

    __instances = {}

    def __new__(cls, form):
        cdef tuple array_form
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
    def array_form(self) -> list:
        return list(self)

    def __call__(self, int el) -> int:
        if not 0 <= el < self.size:
            raise ValueError(f"Called value is out of bounds.")
        return self.array_form[el]

    def __mul__(self, other):
        if not isinstance(other, Permutation):
            raise TypeError(f"Multiplier should be instance of Permutation class, not {type(other)}.")

        cdef list a = self.array_form
        cdef list b = other.array_form
        cdef list perm
        if not b:
            perm = a
        else:
            b.extend(list(range(len(b), len(a))))
            perm = [b[i] for i in a] + b[len(a):]
        return Permutation(perm)

    def __pow__(self, int power, modulo=None):
        if power == -1:
            power = self.order() - 1

        res = Permutation(list(range(self.size)))
        tmp = self
        cdef int i = 1
        while i <= power:
            if i & power:
                res *= tmp
            i <<= 1
            tmp *= tmp
        return res

    @property
    def cyclic_form(self):
        if self._cyclic_form is not None:
            return self._cyclic_form
        cdef list array_form = self.array_form
        cdef list unchecked = [True] * len(array_form)
        cdef list cyclic_form = []
        cdef int i, j
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

    def __repr__(self):
        return f'Permutation({self.array_form})'

    def __str__(self):
        if self.size == 0:
            return '()'
        cdef list cycles = self.cyclic_form
        cdef int big = self.size - 1
        cdef str s = ''
        if not any(i == big for c in cycles for i in c):
            s += '(%s)' % big
        s += ''.join(str(tuple(c)) for c in cycles)
        s = s.replace(',', '')
        return s
