#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 15.01.2021
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

import unittest
from autogrp.automata import *
from autogrp.tools import permute, all_words, captured_output
import numpy as np


TESTS_AMOUNT = 50


class AutomataTestCase(unittest.TestCase):

    def test_orbits(self):
        x = Permutation([1, 2, 0, 3])
        expected = [[0], [1], [2], [3]]
        self.assertListEqual(expected, AutomataGroupElement._get_orbits(x, ALL_FLAGS))
        expected = [[3], [0, 1, 2]]
        self.assertListEqual(expected, AutomataGroupElement._get_orbits(x))

    def test_autogrp_H3(self):
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

    def test_autorgp_H4(self):
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

    def test_lempel_ziv(self):
        H4 = AutomataGroup.generate_H4()

        els_children = []
        els_pemutations = []
        for el in all_words(H4.alphabet, max_len=4):
            el = H4(el)
            els_children.append(el.children)
            els_pemutations.append(el.permutation)

        H4.disable_lempel_ziv()
        for i, el in enumerate(all_words(H4.alphabet, max_len=4)):
            el = H4(el)
            self.assertEqual(els_pemutations[i], el.permutation)
            self.assertEqual(els_children[i], el.children)

    def test_autogrp_is_one(self):

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

    def test_autogrp_is_finite(self):
        H3 = AutomataGroup.generate_H3()
        H4 = AutomataGroup.generate_H4()

        for el in H3.gens + H4.gens:   # type: AutomataGroupElement
            self.assertTrue(el.is_finite())
        self.assertFalse(H4('gcafbgca').is_finite())

    def test_autogrp_order(self):
        H3 = AutomataGroup.generate_H3()
        H4 = AutomataGroup.generate_H4()

        for el in permute(H3.alphabet, repeat=5):
            el = H3(''.join(el))
            if el.is_finite():
                pow_el = el ** el.order()
                self.assertTrue(pow_el.is_one())

        for el in permute(H4.alphabet, repeat=3):
            el = H4(''.join(el))
            if el.is_finite():
                pow_el = el ** el.order()
                self.assertTrue(pow_el.is_one())

        self.assertEqual(2, H4('bab').order())
        self.assertEqual(2, H4('cbabc').order())
        self.assertEqual(2, H4('dcbabcd').order())

    def test_autogrp_is_finite2(self):
        H3 = AutomataGroup.generate_H3()
        H4 = AutomataGroup.generate_H4()

        AutomataGroupElement.disable_cache()

        for el in H3.gens + H4.gens:   # type: AutomataGroupElement
            self.assertEqual(el.is_finite(use_dfs=True),
                             el.is_finite())
        self.assertEqual(H4('gcafbgca').is_finite(use_dfs=True),
                         H4('gcafbgca').is_finite())

        for el in permute(H3.alphabet, repeat=6):
            el = H3(''.join(el))
            self.assertEqual(el.is_finite(use_dfs=True),
                             el.is_finite())

        with self.assertRaises(MaximumOrderDeepError):
            H4('abcfc').is_finite(check_only=0)

    def test_autogrp_is_finite3(self):
        H4 = AutomataGroup.generate_H4()

        AutomataGroupElement.disable_cache()
        for el in all_words(H4.alphabet, max_len=4):
            el = H4(el)
            self.assertEqual(el.is_finite(algo=AS_WORDS),
                             el.is_finite(algo=AS_SHIFTED_WORDS))
            self.assertEqual(el.is_finite(algo=AS_SHIFTED_WORDS),
                             el.is_finite(algo=AS_GROUP_ELEMENTS))

    def test_autogrp_reduce_func(self):
        H4 = AutomataGroup.generate_H4(apply_reduce_func=True)
        orders = []
        for el in all_words(H4.alphabet, max_len=4):
            orders.append(H4(el).order())

        H4 = AutomataGroup.generate_H4(force=True)
        for el, order in zip(all_words(H4.alphabet, max_len=4), orders):
            self.assertEqual(order, H4(el).order())

    def test_autogrp_output(self):
        H4 = AutomataGroup.generate_H4(force=True)
        x = H4('abdfb')

        with captured_output() as (_, _):
            x.enable_cache()
            x.is_finite()

        with captured_output() as (out, err):
            x.disable_cache()
            x.is_finite(verbose=True, print_full_els=True)

        expected = """Generation: 1, element: H4(abdfb = (0 3 2 1) (abf, ab, df, bdb))
Generation: 2, element: H4(abdfbdbabf = (3) (abdbf, dfbab, bdbaf, abfdb))
Generation: 3, element: H4(abdbf = (0 3 2 1) (abf, ab, df, bdb))
Generation: 3, element: H4(dfbab = (0 2 3 1) (b, dfb, fa, dbab))
Found cycle between dfbab and abdfb of length 4.0"""

        self.assertEqual(expected.strip(), out.getvalue().strip())

        with captured_output() as (out, err):
            x.enable_cache()
            x.is_finite(verbose=True, print_full_els=True)
        expected2 = ''
        self.assertEqual(expected2, out.getvalue().strip())

        with captured_output() as (out, err):
            x.disable_cache()
            x.is_finite(verbose=True, print_full_els=False)
        expected3 = """Generation: 1, element: abdfb
Generation: 2, element: abdfbdbabf
Generation: 3, element: abdbf
Generation: 3, element: dfbab
Found cycle between dfbab and abdfb of length 4.0"""
        self.assertEqual(expected3, out.getvalue().strip())


if __name__ == '__main__':
    unittest.main()
