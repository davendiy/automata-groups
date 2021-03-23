#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 23.03.2021
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

import random


def gcd(int x, int y):
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


def permute(space, int repeat, allow_same_neighbours=False) -> str:
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
    cdef str el
    cdef str prev

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


def all_words(space, allow_same_neighbours=False, int max_len=-1):
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
    cdef int i = 1
    cdef str el
    while True:
        for el in permute(space, repeat=i, allow_same_neighbours=allow_same_neighbours):
            yield el
        i += 1

        if max_len != -1 and i > max_len:
            break


def reduce_repetitions(word, atoms):
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
    cdef str tmp = word
    cdef str el
    for el in atoms:
        tmp = tmp.replace(el + el, '')
    while tmp != word:
        word = tmp
        for el in atoms:
            tmp = tmp.replace(el + el, '')
    return tmp


def random_el(space, int repeat, allow_same_neighbours=False):

    if allow_same_neighbours:
        return ''.join(random.choice(space) for _ in range(repeat))

    cdef str first = random.choice(space)
    cdef list res = [first]
    cdef int i, j
    cdef str el
    for _ in range(1, repeat):

        for i, el in enumerate(space):
            if el == res[-1]:
                j = random.randint(0, len(space)-1)
                while j == i:
                    j = random.randint(0, len(space) - 1)
                res.append(space[j])
                break
    return ''.join(res)


def id_func(x):
    return x
