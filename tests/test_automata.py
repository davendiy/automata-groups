#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 15.01.2021
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

import numpy as np

from _autogrp_cython.tools import all_words, captured_output, permute
from autogrp.automata import *

TESTS_AMOUNT = 50


def test_orbits():
    x = Permutation([1, 2, 0, 3])
    expected = [[0], [1], [2], [3]]
    assert expected == AutomataGroupElement._get_orbits(x, ALL_FLAGS)
    expected = [[3], [0, 1, 2]]
    assert expected == AutomataGroupElement._get_orbits(x)


def test_autogrp_H3():
    H3 = AutomataGroup.generate_H3()
    a, b, c = H3.gens
    assert a.name == 'a'
    assert b.name == 'b'
    assert c.name == 'c'

    assert str(a.permutation) == str(Permutation([1, 0, 2]))
    assert str(b.permutation) == str(Permutation([2, 1, 0]))
    assert str(c.permutation) == str(Permutation([0, 2, 1]))

    assert a.children == ('e', 'e', 'a')
    assert b.children == ('e', 'b', 'e')
    assert c.children == ('c', 'e', 'e')

    assert a.parent_group == H3
    assert b.parent_group == H3
    assert c.parent_group == H3

    assert H3.one.name == 'e'


def test_autorgp_H4():
    H4 = AutomataGroup.generate_H4()
    a, b, c, d, f, g = H4.gens
    assert a.name == 'a'
    assert b.name == 'b'
    assert c.name == 'c'
    assert d.name == 'd'
    assert f.name == 'f'
    assert g.name == 'g'

    assert a.children == ('a', 'e', 'e', 'a')
    assert b.children == ('e', 'b', 'e', 'b')
    assert c.children == ('e', 'e', 'c', 'c')
    assert d.children == ('d', 'd', 'e', 'e')
    assert f.children == ('f', 'e', 'f', 'e')
    assert g.children == ('e', 'g', 'g', 'e')

    assert str(a.permutation) == str(Permutation([0, 2, 1, 3]))
    assert str(b.permutation) == str(Permutation([2, 1, 0, 3]))
    assert str(c.permutation) == str(Permutation([1, 0, 2, 3]))
    assert str(d.permutation) == str(Permutation([0, 1, 3, 2]))
    assert str(f.permutation) == str(Permutation([0, 3, 2, 1]))
    assert str(g.permutation) == str(Permutation([3, 1, 2, 0]))

    for el in H4.gens:
        assert el.parent_group == H4

    assert H4.one.name == 'e'

def test_lempel_ziv():
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
        assert els_pemutations[i] == el.permutation
        assert els_children[i] == el.children


def test_autogrp_is_one():

    H3 = AutomataGroup.generate_H3()
    H4 = AutomataGroup.generate_H4()

    for el in H3.gens + H4.gens:
        assert not el.is_one()
        assert (el * el).is_one()

    with open('./interesting_elements/big_non_reducable_trivials.txt') as file:
        els = [line.strip() for line in file]

        for _ in range(TESTS_AMOUNT):
            el = np.random.choice(els)
            assert H3(el).is_one()


def test_autogrp_is_finite():
    H3 = AutomataGroup.generate_H3()
    H4 = AutomataGroup.generate_H4()

    for el in H3.gens + H4.gens:
        assert el.is_finite()
    assert not H4('gcafbgca').is_finite()


def test_autogrp_order():
    H3 = AutomataGroup.generate_H3()
    H4 = AutomataGroup.generate_H4()

    for el in permute(H3.alphabet, repeat=5):
        el = H3(''.join(el))
        if el.is_finite():
            pow_el = el ** el.order()
            assert pow_el.is_one()

    for el in permute(H4.alphabet, repeat=3):
        el = H4(''.join(el))
        if el.is_finite():
            pow_el = el ** el.order()
            assert pow_el.is_one()

    assert 2 == H4('bab').order()
    assert 2 == H4('cbabc').order()
    assert 2 == H4('dcbabcd').order()


def test_autogrp_is_finite2():
    H3 = AutomataGroup.generate_H3()
    H4 = AutomataGroup.generate_H4()

    H4.disable_cache()
    H3.disable_cache()

    for el in H3.gens + H4.gens:
        assert (el.is_finite(use_dfs=True),
                            el.is_finite())
    assert (H4('gcafbgca').is_finite(use_dfs=True),
                        H4('gcafbgca').is_finite())

    for el in permute(H3.alphabet, repeat=6):
        el = H3(''.join(el))
        assert (el.is_finite(use_dfs=True),
                            el.is_finite())

    try:
        H4('abcfc').is_finite(check_only=0)
    except MaximumOrderDeepError:
        pass
    else:
        assert False


def test_autogrp_is_finite3():
    H4 = AutomataGroup.generate_H4()

    H4.disable_cache()
    for el in all_words(H4.alphabet, max_len=4):
        el = H4(el)
        assert (el.is_finite(algo=AS_WORDS),
                            el.is_finite(algo=AS_SHIFTED_WORDS))
        assert (el.is_finite(algo=AS_SHIFTED_WORDS),
                            el.is_finite(algo=AS_GROUP_ELEMENTS))


def test_autogrp_reduce_func():
    H4 = AutomataGroup.generate_H4(apply_reduce_func=True)
    orders = []
    for el in all_words(H4.alphabet, max_len=4):
        orders.append(H4(el).order())

    H4 = AutomataGroup.generate_H4(force=True)
    for el, order in zip(all_words(H4.alphabet, max_len=4), orders):
        assert order == H4(el).order()


def test_call():
    H4 = AutomataGroup.generate_H4()
    assert '11111' == H4.one('11111')
    assert '21111' == H4('a')(11111)
    assert '12222' == H4('a')(22222)
    assert '01222' == H4('ab')(22222)

    H3 = AutomataGroup.generate_H3()
    assert '0000' == H3('abcabcabcabcabca')('2222')


def test_autogrp_output():
    H4 = AutomataGroup.generate_H4(force=True)
    x = H4('abdfb')

    with captured_output() as (_, _):
        x.is_finite()

    with captured_output() as (out, err):
        H4.disable_cache()
        x.is_finite(verbose=True, print_full_els=True)

    expected = """Generation: 0, element: H4(abdfb = (0 3 2 1) (abf, ab, df, bdb))
Generation: 1, element: H4(abdfbdbabf = (3) (abdbf, dfbab, bdbaf, abfdb))
Generation: 2, element: H4(abdbf = (0 3 2 1) (abf, ab, df, bdb))
Generation: 2, element: H4(dfbab = (0 2 3 1) (b, dfb, fa, dbab))
Found cycle between dfbab and abdfb of length 4.0"""

    assert expected.strip() == out.getvalue().strip()

    with captured_output() as (out, err):
        H4.enable_cache()
        x.is_finite(verbose=True, print_full_els=True)
    expected2 = ''
    assert expected2 == out.getvalue().strip()

    with captured_output() as (out, err):
        H4.disable_cache()
        x.is_finite(verbose=True, print_full_els=False)
    expected3 = """Generation: 0, element: abdfb
Generation: 1, element: abdfbdbabf
Generation: 2, element: abdbf
Generation: 2, element: dfbab
Found cycle between dfbab and abdfb of length 4.0"""
    assert expected3 == out.getvalue().strip()
