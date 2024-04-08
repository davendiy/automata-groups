
from autogrp import free_reduce, reduce_repetitions, as_tokens


def test_free_reduce():
    word = 'aba^-2bb^-1a^2bba'
    assert free_reduce(word) == 'abbba'
    assert free_reduce(word, use_powering=True) == 'ab^3a'
    assert as_tokens(free_reduce(word, use_powering=True), split=True) == ['a', 'b', 'b', 'b', 'a']

    word = 's1s2s1s1s1s2^-1s1^3'
    assert free_reduce(word, alphabet=['s1', 's2']) == 's1s2s1s1s1s2^-1s1s1s1'
    assert free_reduce(word, alphabet=['s1', 's2'], use_powering=True) == 's1s2s1^3s2^-1s1^3'

    try:
        free_reduce(word)
    except ValueError:
        assert True
    else:
        assert False


def test_repetitions():

    word = 'aba^-2bb^-1a^2bba'
    assert reduce_repetitions(word) == 'aba'
    assert reduce_repetitions(word, use_powering=True) == 'aba'

    word = 's1s2s1s1s1s2^-1s1^3'
    assert reduce_repetitions(word, alphabet=['s1', 's2']) == 's1s2s1s2s1'

    try:
        reduce_repetitions(word)
    except ValueError:
        assert True
    else:
        assert False
