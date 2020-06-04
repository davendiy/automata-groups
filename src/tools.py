#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 17.04.2020
# Excusa. Quod scripsi, scripsi.

# by David Zashkolny
# email: davendiy@gmail.com

from .source2 import *


def random_el(n):
    space = ['a', 'b', 'c']
    first = np.random.choice(space)
    res = [first]
    for i in range(1, n):
        if res[i-1] == 'a':

            res.append(np.random.choice(['b', 'c']))
        elif res[i-1] == 'b':
            res.append(np.random.choice(['a', 'c']))
        else:
            res.append(np.random.choice(['a', 'b']))
    return ''.join(res)


def permute(seq, repeat):
    if repeat == 1:
        for el in seq:
            yield [el]
    elif repeat < 1:
        yield []
    else:
        for prev in permute(seq, repeat-1):
            for el in seq:
                if prev[-1] == el:
                    continue
                yield prev + [el]
