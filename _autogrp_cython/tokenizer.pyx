
import re
from typing import Iterable
from dataclasses import dataclass
from string import ascii_lowercase

from _autogrp_cython.trie import Trie

"""
TODO:
 - use Cython features
 - implement free group reduction as pipe
 - implement Hanoi reduction as pipe
 - rethink the entire architecture, work with Tokens
"""

from itertools import chain

degree = re.compile(r'\^-?\d*')


@dataclass(slots=True, frozen=True)
class Token:
    el: str
    degree: int

    def split(self):
        if abs(self.degree) > 1:
            sign = 1 if self.degree > 0 else -1
            return [Token(self.el, sign) for _ in range(abs(self.degree))]
        else:
            return [self]

    def get_repr(self, use_powering=True):
        if self.degree == 1:
                return self.el
        elif self.degree == 0:
            return ''

        if use_powering:
            return f'{self.el}^{self.degree}'

        res = self.el if self.degree > 0 else f'{self.el}^-1'
        return res * abs(self.degree)


def tokenize(word, alphabet: Iterable[str] = None) -> Iterable[Token]:
    """Tokenize the given word over the alphabet. If alphabet is absent,
    assume it is an english one.

    Each token is a pair `(el, degree)` where `el` represents an element from
    the alphabet and `degree` is an integer.

    >>> list(tokenize('aba^-1 b^-1c ba^1231 b^-123', alphabet='abc'))
    [Token(el='a', degree=1), Token(el='b', degree=1), Token(el='a', degree=-1), Token(el='b', degree=-1), Token(el='c', degree=1), Token(el='b', degree=1), Token(el='a', degree=1231), Token(el='b', degree=-123)]
    >>> list(tokenize('aba^-1 b^-1c b1a1b_aa^-1aa^1231 b^-123', alphabet=['a', 'b', 'c', 'b1', 'a1', 'b_a']))
    [Token(el='a', degree=1), Token(el='b', degree=1), Token(el='a', degree=-1), Token(el='b', degree=-1), Token(el='c', degree=1), Token(el='b1', degree=1), Token(el='a1', degree=1), Token(el='b_a', degree=1), Token(el='a', degree=-1), Token(el='a', degree=1), Token(el='a', degree=1231), Token(el='b', degree=-123)]
    >>> list(tokenize('abbab^2342-', alphabet='ab'))
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/Users/mac/Documents/univer/phd/hanoi_groups/automata-groups/autogrp/tokenizer.py", line 38, in tokenize
        raise ValueError(f"unknown prefix: {word}")
    ValueError: unknown prefix: -
    """
    if alphabet is None:
        alphabet = ascii_lowercase
    alphabet = Trie(*alphabet)
    cdef int end
    cdef str cur_el
    cdef int cur_degree

    while word:
        word = word.strip()
        cur_el, word = alphabet.max_prefix(word)
        if not cur_el:
            raise ValueError(f"unknown prefix: {word}")
        match = degree.match(word)
        if match is not None:
            end = match.span()[1]
            cur_degree = int(word[1:end])
            word = word[end:]
        else:
            cur_degree = 1
        yield Token(cur_el, cur_degree)


def as_tokens(word, alphabet: Iterable[str] = None, split=False) -> Iterable[str]:
    if split:
        seq = [tok.split() for tok in tokenize(word, alphabet)]
    else:
        seq = [tokenize(word, alphabet)]

    return [el.get_repr() for el in chain(*seq)]
