
from collections import deque
from typing import Iterable
from _autogrp_cython.tokenizer import Token, tokenize


# FIXME: too strong code smell
def _reduce_tokens_free(tok1: Token, tok2: Token = None):
    if tok2 is None:
        return [] if tok1.degree == 0 else [tok1]

    if tok1.el == tok2.el:
        res = Token(tok1.el, tok1.degree + tok2.degree)
        return _reduce_tokens_free(res)
    else:
        return _reduce_tokens_free(tok1) + _reduce_tokens_free(tok2)


def _reduce_tokens_repetition(tok1: Token, tok2: Token = None):
    if tok2 is None:
        return [] if tok1.degree % 2 == 0 else [Token(tok1.el, 1)]

    if tok1.el == tok2.el:
        res = Token(tok1.el, tok1.degree + tok2.degree)
        return _reduce_tokens_repetition(res)
    else:
        return _reduce_tokens_repetition(tok1) + _reduce_tokens_repetition(tok2)


def reduce_pipe(seq: Iterable[Token], reduce_func):
    stack = deque()
    for tok in seq:
        if not stack:
            stack.extend(reduce_func(tok))
            continue

        prev = stack.pop()
        stack.extend(reduce_func(prev, tok))
    return stack


def recover_word(seq: Iterable[Token], use_powering=False):
    return ''.join(tok.get_repr(use_powering=use_powering) for tok in seq)


def free_reduce(word, alphabet: Iterable[str] = None, use_powering=False):
    seq = tokenize(word, alphabet)
    seq = reduce_pipe(seq, _reduce_tokens_free)
    return recover_word(seq, use_powering=use_powering)


def reduce_repetitions(word, alphabet: Iterable[str] = None, use_powering=False):
    seq = tokenize(word, alphabet)
    seq = reduce_pipe(seq, _reduce_tokens_repetition)
    return recover_word(seq, use_powering=use_powering)
