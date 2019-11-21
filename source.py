#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 22.11.2019
# by David Zashkolny
# 3 course, comp math
# Taras Shevchenko National University of Kyiv
# email: davendiy@gmail.com


from sympy.combinatorics import Permutation
from numpy import lcm

CHUNK = 400

RECURSION_MAX_DEEP = 20


class NotCalculatedError(Exception):
    pass


class BadExpressionError(Exception):
    pass


class NewGroupElement:
    DEFINED_ELEMENTS = {"e": ...}

    @staticmethod
    def from_cache(name):
        if name in NewGroupElement.DEFINED_ELEMENTS:
            return NewGroupElement.DEFINED_ELEMENTS[name]
        else:
            raise NotCalculatedError()

    def __init__(self, name="e", perm="", el_list=("e", "e", "e"), primitive=True):
        self.prim = primitive

        if self.prim:
            self.name = "e"
            self.perm = Permutation([0, 1, 2])
            self.el_list = ("e", "e", "e")
            NewGroupElement.DEFINED_ELEMENTS['e'] = self

        else:
            if name in NewGroupElement.DEFINED_ELEMENTS:
                tmp = NewGroupElement.DEFINED_ELEMENTS[name]
                self.name = tmp.name
                self.perm = tmp.perm
                self.el_list = tmp.el_list
            else:
                assert isinstance(perm, Permutation), "bad type of permutation"
                assert isinstance(el_list, list) or isinstance(el_list, tuple), "bad type of el_list"
                assert len(el_list) == 3, "bad lenght of el_list"
                assert isinstance(name, str), "bad name"
                for el in el_list:
                    if el not in NewGroupElement.DEFINED_ELEMENTS and el != name:
                        NewGroupElement.parse_str(el)

                self.name = name
                self.perm = perm
                self.el_list = el_list

                NewGroupElement.DEFINED_ELEMENTS[self.name] = self

                if self.is_primitive():
                    self.perm = Permutation([0, 1, 2])
                    self.el_list = ("e", "e", "e")
                    self.prim = True
                    NewGroupElement.DEFINED_ELEMENTS[self.name] = self

    def __str__(self):
        return self.name + " = " + str(self.perm) + " (" + ', '.join(self.el_list) + ")"

    def __repr__(self):
        return self.name

    def __call__(self, word):
        if self.prim:
            return word

        elif not word:
            return []
        elif len(word) == 1:
            return [word[0] ^ self.perm]
        else:
            el = self.el_list[int(word[0]) - 1]
            return [word[0] ^ self.perm] + NewGroupElement.DEFINED_ELEMENTS[el](word[1:])

    def __mul__(self, other):
        if self.prim:
            return other

        if other.prim:
            return self

        res_name = self.name + other.name
        res_perm = other.perm * self.perm

        res_els = []
        for i in range(3):
            tmp1 = self.el_list[i ^ other.perm]
            tmp2 = other.el_list[i]
            tmp_res = tmp1 + tmp2
            tmp_res = tmp_res.replace('aa', '').replace('bb', '').replace('cc', '').replace('e', '')
            tmp_res = tmp_res if tmp_res else "e"
            res_els.append(tmp_res)

        for el in res_els:
            NewGroupElement.parse_str(el)
        res = NewGroupElement(res_name, res_perm, res_els, primitive=False)
        return res

    def __pow__(self, power):
        res = NewGroupElement()
        for i in range(power):
            res *= self
        return res

    def is_primitive(self, checked=()):
        if not checked:
            checked = set()
        if self.name in checked:
            return True
        if self.prim:
            return self.prim

        if self.perm != Permutation([0, 1, 2]):
            return False

        succ = True
        checked.add(self.name)
        for el in self.el_list:
            tmp = NewGroupElement.DEFINED_ELEMENTS[el]
            succ = succ and tmp.is_primitive(checked)
            if not succ:
                break
        return succ

    def order(self):
        if self.prim:
            return 1

        perm_order = self.perm.order()
        tmp = self ** perm_order
        if (tmp.el_list[0] == tmp.el_list[0][::-1]
                and tmp.el_list[1] == tmp.el_list[1][::-1]
                and tmp.el_list[2] == tmp.el_list[2][::-1]):
            return perm_order * 2 if not tmp.prim else perm_order
        else:
            return float('inf')

    def order2(self):
        checked = set()
        tmp_res = self._order2_helper(checked, 0)
        return tmp_res if tmp_res else float('inf')

    def _order2_helper(self, checked, deep):
        if self.name in checked and not self.prim:
            return 0
        elif self.prim:
            return 1
        elif deep > RECURSION_MAX_DEEP:
            return 0

        res = self.perm.order()
        tmp = self ** res

        checked.add(self.name)

        lcm_ord = 1
        for el in tmp.el_list:
            tmp = NewGroupElement.parse_str(el)
            tmp_ord = tmp._order2_helper(checked, deep + 1)
            lcm_ord = lcm(lcm_ord, tmp_ord)
            if not lcm_ord:
                break
        return res * lcm_ord

    def order_brute(self):
        tmp = NewGroupElement()
        res = float('inf')
        for i in range(CHUNK):
            tmp *= self
            if tmp.prim:
                res = i
                break
        return res

    @staticmethod
    def parse_str(expression):
        if expression in NewGroupElement.DEFINED_ELEMENTS:
            return NewGroupElement.DEFINED_ELEMENTS[expression]

        for el in set(expression):
            if el not in NewGroupElement.DEFINED_ELEMENTS:
                raise BadExpressionError()

        res = NewGroupElement()
        for el in expression:
            res = res * NewGroupElement.from_cache(el)
        return res


def mul(tmp1, tmp2):
    res = tmp1 * tmp2
    print(res.get_full_str())
    return res


e = NewGroupElement()
a = NewGroupElement(name='a', perm=Permutation([1, 0, 2]), el_list=['e', 'e', 'a'], primitive=False)
b = NewGroupElement(name='b', perm=Permutation([2, 1, 0]), el_list=['e', 'b', 'e'], primitive=False)
c = NewGroupElement(name='c', perm=Permutation([0, 2, 1]), el_list=['c', 'e', 'e'], primitive=False)


if __name__ == '__main__':

    print(a, b, c, sep='\n')
    test = NewGroupElement.parse_str("abcbabcbacba")
    print(test, test.order())
