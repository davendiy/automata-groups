#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 17.04.2020
# Excusa. Quod scripsi, scripsi.

# by David Zashkolny
# email: davendiy@gmail.com

import random

import sys
from contextlib import contextmanager
from io import StringIO
import time


@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def gcd(x: int, y: int):
    """ Classical gcd for two integers.
    """
    while y:
        x, y = y, x % y
    return x


def lcm(x, y):
    """ Lcm of two integers that supports float('inf') value.

    Returns
    -------
    lcm of (x, y) or float('inf') if one of them is float('inf')
    """
    if x == float('inf') or y == float('inf'):
        return float('inf')
    else:
        return x * y // gcd(x, y)


def permute(space, repeat: int, allow_same_neighbours=False) -> str:
    """ Get all the words of letters from the given
    space of given length.

    Parameters
    ----------
    space   : Iterable container of str elements of len 1.
    repeat  : int, length of yielded words
    allow_same_neighbours : if False then all words that don't have same elements
                            on the neighbour positions will be returned and all
                            possible words otherwise.
                            Default False
    Yields
    -------
    str, result word
    """
    if repeat == 1:
        for el in space:
            yield el
    elif repeat < 1:
        yield ''
    else:
        for prev in permute(space, repeat - 1, allow_same_neighbours=allow_same_neighbours):
            for el in space:
                if prev[-1] == el and not allow_same_neighbours:
                    continue
                yield prev + el


def all_words(space, allow_same_neighbours=False, max_len=None):
    """ Get all possible words (infinite set) of elements from the given
    space.

    Parameters
    ----------
    space   : Iterable container of str elements of len 1.
    allow_same_neighbours : if False then all words that don't have same elements
                            on the neighbour positions will be returned and all
                            possible words otherwise.
                            Default False.
    max_len : maximum allowed length of returned words.
              Default None, which means that there is no maximum length

    Yields
    -------
    str, result word
    """
    i = 1
    while True:
        for el in permute(space, repeat=i, allow_same_neighbours=allow_same_neighbours):
            yield el
        i += 1

        if max_len is not None and i > max_len:
            break


def reduce_repetitions(word: str, atoms):
    """ Reduce given word assuming that given atoms have order 2
    (i.e. repetition of such atoms of length 2 like 'aa' means empty word).

    Parameters
    ----------
    word   : str, word of any length
    atoms  : iterable of elements that could appear in word of order 2

    Returns
    -------
    Reduced word

    Examples
    --------
    >>> reduce_repetitions('caaaabbbbc', ['a', 'b', 'c'])
    ''
    >>> reduce_repetitions('caaaabbbbc', ['a', 'b'])
    'cc'
    """
    tmp = word
    for el in atoms:
        tmp = tmp.replace(el + el, '')
    while tmp != word:
        word = tmp
        for el in atoms:
            tmp = tmp.replace(el + el, '')
    return tmp


def random_el(space, repeat, allow_same_neighbours=False):

    if allow_same_neighbours:
        return ''.join(random.choice(space) for _ in range(repeat))

    first = random.choice(space)
    res = [first]
    for _ in range(1, repeat):

        for i, el in enumerate(space):
            if el == res[-1]:
                while (j := random.randint(0, len(space)-1)) == i:
                    pass
                res.append(space[j])
                break
    return ''.join(res)


def id_func(x):
    return x


def do_each(n):
    def _do_each_n(func):
        j = 0

        def _func(*args, **kwargs):
            nonlocal j
            j += 1
            if j == n:
                j = 0
                return func(*args, **kwargs)
        return _func
    return _do_each_n


def info(cur_amount, full_amount, start_time=None):

    print('\r', end='')
    print('done: {:.4f}%'.format(float(cur_amount / full_amount) * 100), end='')

    if start_time is not None:
        cur_time = time.time()
        elapsed = cur_time - start_time
        on_clock = elapsed / cur_amount
        hyp_full_time = on_clock * full_amount
        remaind = hyp_full_time - elapsed

        if elapsed > 3600: elapsed_str = '{:.2f}h'.format(elapsed / 3600)
        elif elapsed > 60: elapsed_str = '{:.2f}m'.format(elapsed / 60)
        else:              elapsed_str = '{:.2f}s'.format(elapsed)

        if remaind > 3600: remaind_str = '{:.2f}h'.format(remaind / 3600)
        elif remaind > 60: remaind_str = '{:.2f}m'.format(remaind / 60)
        else:              remaind_str = '{:.2f}s'.format(remaind)

        print(f'   time passed: {elapsed_str},', end='')
        print(f'   approximately left: {remaind_str}      ', end='')


class DiGraph:

    def __init__(self):
        pass

    def add_vertex(self, vert):
        pass

    def add_edge(self, source, dest, value):
        pass
