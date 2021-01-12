#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 13.11.2019
# Excusa. Quod scripsi, scripsi.

# by David Zashkolny
# email: davendiy@gmail.com

from sympy.combinatorics import Permutation
from numpy import lcm
from multiprocessing import RLock

LOCK = RLock()

CHUNK = 400

RECURSION_MAX_DEEP = 20


class NotCalculatedError(Exception):
    pass


class BadExpressionError(Exception):
    pass


class AutomataGroupElement:
    DEFINED_ELEMENTS = {"e": ...}

    @staticmethod
    def from_cache(name):
        if name in AutomataGroupElement.DEFINED_ELEMENTS:
            return AutomataGroupElement.DEFINED_ELEMENTS[name]
        else:
            raise NotCalculatedError()

    def __init__(self, name="e", perm="", el_list=("e", "e", "e"), primitive=True):
        self.triv = primitive

        if self.triv:
            self.name = "e"
            self.perm = Permutation([0, 1, 2])
            self.el_list = ("e", "e", "e")
            with LOCK:
                AutomataGroupElement.DEFINED_ELEMENTS['e'] = self

        else:
            if name in AutomataGroupElement.DEFINED_ELEMENTS:
                tmp = AutomataGroupElement.DEFINED_ELEMENTS[name]
                self.name = tmp.name
                self.perm = tmp._perm
                self.el_list = tmp.el_list
            else:
                assert isinstance(perm, Permutation), "bad type of permutation"
                assert isinstance(el_list, list) or isinstance(el_list, tuple), "bad type of el_list"
                assert len(el_list) == 3, "bad lenght of el_list"
                assert isinstance(name, str), "bad name"
                for el in el_list:
                    if el not in AutomataGroupElement.DEFINED_ELEMENTS and el != name:
                        GroupElement(el)

                self.name = name
                self.perm = perm
                self.el_list = el_list

                with LOCK:
                    AutomataGroupElement.DEFINED_ELEMENTS[self.name] = self

                # if self.is_primitive():
                #     self.perm = Permutation([0, 1, 2])
                #     self.el_list = ("e", "e", "e")
                #     self.prim = True
                #     with LOCK:
                #         AutomataGroupElement.DEFINED_ELEMENTS[self.name] = self

    def __str__(self):
        return self.name + " = " + str(self.perm) + " (" + ', '.join(self.el_list) + ")"

    def __repr__(self):
        return self.name

    def __call__(self, word):
        if self.triv:
            return word

        elif not word:
            return []
        elif len(word) == 1:
            return [word[0] ^ self.perm]
        else:
            el = self.el_list[int(word[0]) - 1]
            return [word[0] ^ self.perm] + AutomataGroupElement.DEFINED_ELEMENTS[el](word[1:])

    def __mul__(self, other):
        if self.triv:
            return other

        if other.triv:
            return self

        res_name = self.name + other.name
        res_perm = other._perm * self.perm

        res_els = []
        for i in range(3):
            tmp1 = self.el_list[i ^ other._perm]
            tmp2 = other.el_list[i]
            tmp_res = tmp1 + tmp2
            tmp_res = tmp_res.replace('aa', '').replace('bb', '').replace('cc', '').replace('e', '')
            tmp_res = tmp_res if tmp_res else "e"
            res_els.append(tmp_res)

        for el in res_els:
            GroupElement(el)
        res = AutomataGroupElement(res_name, res_perm, res_els, primitive=False)
        return res

    def __pow__(self, power):
        res = AutomataGroupElement()
        for i in range(power):
            res *= self
        return res

    def is_trivial(self, checked=()):
        if not checked:
            checked = set()
        if self.name in checked:
            return True
        if self.triv:
            return self.triv

        if self.perm != Permutation([0, 1, 2]):
            return False

        succ = True
        checked.add(self.name)
        for el in self.el_list:
            tmp = AutomataGroupElement.DEFINED_ELEMENTS[el]
            succ = succ and tmp.is_trivial(checked)
            if not succ:
                break
        return succ

    def order(self):
        if self.triv:
            return 1

        perm_order = self.perm.order()
        tmp = self ** perm_order
        if (tmp.el_list[0] == tmp.el_list[0][::-1]
                and tmp.el_list[1] == tmp.el_list[1][::-1]
                and tmp.el_list[2] == tmp.el_list[2][::-1]):
            return perm_order * 2 if not tmp.triv else perm_order
        else:
            return float('inf')

    def order2(self):
        checked = set()
        tmp_res = self._order2_helper(checked, 0)
        return tmp_res if tmp_res else float('inf')

    def _order2_helper(self, checked, deep):
        if self.name in checked and not self.triv:
            return 0
        elif self.triv:
            return 1
        elif deep > RECURSION_MAX_DEEP:
            return 0

        res = self.perm.order()
        tmp = self ** res

        checked.add(self.name)

        lcm_ord = 1
        for el in tmp.el_list:
            tmp = GroupElement(el)
            tmp_ord = tmp._order2_helper(checked, deep + 1)
            lcm_ord = lcm(lcm_ord, tmp_ord)
            if not lcm_ord:
                break
        return res * lcm_ord

    def order_brute(self):
        tmp = AutomataGroupElement()
        res = float('inf')
        for i in range(CHUNK):
            tmp *= self
            if tmp.triv:
                res = i
                break
        return res


def GroupElement(expression):
    if expression in AutomataGroupElement.DEFINED_ELEMENTS:
        return AutomataGroupElement.DEFINED_ELEMENTS[expression]

    for el in set(expression):
        if el not in AutomataGroupElement.DEFINED_ELEMENTS:
            raise BadExpressionError()

    res = AutomataGroupElement()
    for el in expression:
        res = res * AutomataGroupElement.from_cache(el)
    return res


def mul(tmp1, tmp2):
    res = tmp1 * tmp2
    print(res.get_full_str())
    return res


e = AutomataGroupElement()
a = AutomataGroupElement(name='a', perm=Permutation([1, 0, 2]), el_list=['e', 'e', 'a'], primitive=False)
b = AutomataGroupElement(name='b', perm=Permutation([2, 1, 0]), el_list=['e', 'b', 'e'], primitive=False)
c = AutomataGroupElement(name='c', perm=Permutation([0, 2, 1]), el_list=['c', 'e', 'e'], primitive=False)


if __name__ == '__main__':

    print(a, b, c, sep='\n')
    test = GroupElement("abcbabcbacba")
    print(test, test.order())
