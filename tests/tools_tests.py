#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 15.01.2021
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

import unittest
from autogrp.tools import *
import numpy as np

TESTS_AMOUNT = 50


class MyTestCase(unittest.TestCase):

    def test_gcd(self):
        for _ in range(TESTS_AMOUNT):
            a = np.random.randint(0, 100)
            b = np.random.randint(0, 100)
            self.assertEqual(gcd(a, b), np.gcd(a, b))

    def test_lcm(self):
        for _ in range(TESTS_AMOUNT):
            a = np.random.randint(0, 100)
            b = np.random.randint(0, 100)
            self.assertEqual(lcm(a, b), np.lcm(a, b))
        self.assertEqual(lcm(10, float('inf')), float('inf'))

    def test_reduce_repetitions(self):
        test_str = 'aaaabbbbaaaa'
        self.assertEqual('', reduce_repetitions(test_str, ['a', 'b']))
        test_str = 'aaaabbbbaaa'
        self.assertEqual('a', reduce_repetitions(test_str, ['a', 'b']))
        test_str = 'dcccaaaabbbbaaaacccd'
        self.assertEqual('', reduce_repetitions(test_str, ['a', 'b', 'c', 'd']))


if __name__ == '__main__':
    unittest.main()
