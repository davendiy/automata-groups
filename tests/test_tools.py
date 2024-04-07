#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 15.01.2021
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

import numpy as np

from _autogrp_cython.tools import gcd, lcm, reduce_repetitions

TESTS_AMOUNT = 50


def test_gcd():
    for _ in range(TESTS_AMOUNT):
        a = np.random.randint(0, 100)
        b = np.random.randint(0, 100)
        assert gcd(a, b) == np.gcd(a, b)


def test_lcm():
    for _ in range(TESTS_AMOUNT):
        a = np.random.randint(0, 100)
        b = np.random.randint(0, 100)
        assert lcm(a, b) == np.lcm(a, b)
    assert lcm(10, float('inf')) == float('inf')


def test_reduce_repetitions():
    test_str = 'aaaabbbbaaaa'
    assert '' == reduce_repetitions(test_str, ['a', 'b'])
    test_str = 'aaaabbbbaaa'
    assert 'a' == reduce_repetitions(test_str, ['a', 'b'])
    test_str = 'dcccaaaabbbbaaaacccd'
    assert '' == reduce_repetitions(test_str, ['a', 'b', 'c', 'd'])
