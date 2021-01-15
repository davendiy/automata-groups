#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 15.01.2021
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

import unittest
from src.source3 import *
import numpy as np
from sympy.combinatorics import Permutation

TESTS_AMOUNT = 50


class MyTestCase(unittest.TestCase):

    def test_H3(self):
        H3 = AutomataGroup.generate_H3()
        a, b, c = H3.gens    # type: AutomataGroupElement
        self.assertEqual(a.name, 'a')
        self.assertEqual(b.name, 'b')
        self.assertEqual(c.name, 'c')

        self.assertEqual(str(a.permutation), str(Permutation([1, 0, 2])))
        self.assertEqual(str(b.permutation), str(Permutation([2, 1, 0])))
        self.assertEqual(str(c.permutation), str(Permutation([0, 2, 1])))

        self.assertEqual(a.children, ('e', 'e', 'a'))
        self.assertEqual(b.children, ('e', 'b', 'e'))
        self.assertEqual(c.children, ('c', 'e', 'e'))

        self.assertEqual(a.parent_group, H3)
        self.assertEqual(b.parent_group, H3)
        self.assertEqual(c.parent_group, H3)

        self.assertEqual(H3.one.name, 'e')

    def test_H4(self):
        H4 = AutomataGroup.generate_H4()
        a, b, c, d, f, g = H4.gens   # type: AutomataGroupElement
        self.assertEqual(a.name, 'a')
        self.assertEqual(b.name, 'b')
        self.assertEqual(c.name, 'c')
        self.assertEqual(d.name, 'd')
        self.assertEqual(f.name, 'f')
        self.assertEqual(g.name, 'g')

        self.assertEqual(a.children, ('a', 'e', 'e', 'a'))
        self.assertEqual(b.children, ('e', 'b', 'e', 'b'))
        self.assertEqual(c.children, ('e', 'e', 'c', 'c'))
        self.assertEqual(d.children, ('d', 'd', 'e', 'e'))
        self.assertEqual(f.children, ('f', 'e', 'f', 'e'))
        self.assertEqual(g.children, ('e', 'g', 'g', 'e'))

        self.assertEqual(str(a.permutation), str(Permutation([0, 2, 1, 3])))
        self.assertEqual(str(b.permutation), str(Permutation([2, 1, 0, 3])))
        self.assertEqual(str(c.permutation), str(Permutation([1, 0, 2, 3])))
        self.assertEqual(str(d.permutation), str(Permutation([0, 1, 3, 2])))
        self.assertEqual(str(f.permutation), str(Permutation([0, 3, 2, 1])))
        self.assertEqual(str(g.permutation), str(Permutation([3, 1, 2, 0])))

        for el in H4.gens:
            self.assertEqual(el.parent_group, H4)

        self.assertEqual(H4.one.name, 'e')

    def test_is_one(self):

        H3 = AutomataGroup.generate_H3()
        H4 = AutomataGroup.generate_H4()

        for el in H3.gens + H4.gens:   # type: AutomataGroupElement
            self.assertFalse(el.is_one())
            self.assertTrue((el * el).is_one())

        with open('../interesting_elements/big_non_reducable_trivials.txt') as file:
            els = [line.strip() for line in file]

            for _ in range(TESTS_AMOUNT):
                el = np.random.choice(els)
                self.assertTrue(H3(el).is_one())

    def test_is_finite(self):
        H3 = AutomataGroup.generate_H3()
        H4 = AutomataGroup.generate_H4()

        for el in H3.gens + H4.gens:   # type: AutomataGroupElement
            self.assertTrue(el.is_finite())
        self.assertFalse(H4('gcafbgca').is_finite())

    def test_order(self):
        pass

if __name__ == '__main__':
    unittest.main()
