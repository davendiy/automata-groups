#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 15.01.2021
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

from autogrp import Permutation as my_Permutation
from sympy.combinatorics import Permutation as sym_Permutation
import numpy as np

MAX_LEN = 20
TESTS_AMOUNT = 100


def test_perm_str():
    for _ in range(TESTS_AMOUNT):
        length = np.random.randint(MAX_LEN)
        perm = list(range(length))
        np.random.shuffle(perm)
        my = my_Permutation(perm)
        sym = sym_Permutation(perm)
        assert str(sym) == str(my)


def test_perm_cyclic_form():
    for _ in range(TESTS_AMOUNT):
        length = np.random.randint(MAX_LEN)
        perm = list(range(length))
        np.random.shuffle(perm)
        my = my_Permutation(perm)
        sym = sym_Permutation(perm)
        assert sym.cyclic_form == my.cyclic_form


def test_perm_mul():
    for _ in range(TESTS_AMOUNT):
        length1 = np.random.randint(MAX_LEN)
        length2 = np.random.randint(MAX_LEN)
        perm1 = list(range(length1))
        perm2 = list(range(length2))
        np.random.shuffle(perm1)
        np.random.shuffle(perm2)

        my1 = my_Permutation(perm1)
        my2 = my_Permutation(perm2)
        my_res = my1 * my2

        sym1 = sym_Permutation(perm1)
        sym2 = sym_Permutation(perm2)
        sym_res = sym1 * sym2
        assert sym_res.cyclic_form == my_res.cyclic_form


def test_perm_call():
    for _ in range(TESTS_AMOUNT):
        length = np.random.randint(MAX_LEN)
        perm = list(range(length))
        np.random.shuffle(perm)

        my = my_Permutation(perm)
        sym = sym_Permutation(perm)

        for i in range(length):
            assert sym(i) == my(i)


def test_perm_size():
    for _ in range(TESTS_AMOUNT):
        length = np.random.randint(MAX_LEN)
        perm = list(range(length))
        np.random.shuffle(perm)
        my = my_Permutation(perm)
        sym = sym_Permutation(perm)
        assert sym.size == my.size


def test_perm_order():
    for _ in range(TESTS_AMOUNT):
        length = np.random.randint(MAX_LEN)
        perm = list(range(length))
        np.random.shuffle(perm)
        my = my_Permutation(perm)
        sym = sym_Permutation(perm)
        assert sym.order() == my.order()


def test_power():
    for _ in range(TESTS_AMOUNT):
        length = np.random.randint(MAX_LEN)
        perm = list(range(length))
        np.random.shuffle(perm)
        my = my_Permutation(perm)
        sym = sym_Permutation(perm)
        power = np.random.randint(MAX_LEN * 4)
        assert str(sym ** power) == str(my ** power)
