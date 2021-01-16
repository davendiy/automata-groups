#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 17.04.2020
# Excusa. Quod scripsi, scripsi.

# by David Zashkolny
# email: davendiy@gmail.com


def gcd(x, y):
    while y:
        x, y = y, x % y
    return x


def lcm(x, y):
    if x == float('inf') or y == float('inf'):
        return float('inf')
    else:
        return x * y // gcd(x, y)


def permute(space, repeat, allow_same_neighbours=False):
    if repeat == 1:
        for el in space:
            yield [el]
    elif repeat < 1:
        yield []
    else:
        for prev in permute(space, repeat - 1, allow_same_neighbours=allow_same_neighbours):
            for el in space:
                if prev[-1] == el and not allow_same_neighbours:
                    continue
                yield prev + [el]


def all_words(space, allow_same_neighbours=False):
    i = 1
    while True:
        for el in permute(space, repeat=i, allow_same_neighbours=allow_same_neighbours):
            yield ''.join(el)
        i += 1


def reduce_repetitions(word: str, atoms):
    tmp = word
    for el in atoms:
        tmp = tmp.replace(el + el, '')
    while tmp != word:
        word = tmp
        for el in atoms:
            tmp = tmp.replace(el + el, '')
    return tmp


def id_func(x):
    return x
