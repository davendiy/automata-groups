
from collections import deque
from typing import Iterable
from _autogrp_cython.tokenizer import Token, tokenize



def _add_tokens_free(tok1: Token, tok2: Token): 
    res = Token(tok1.el, tok1.degree + tok2.degree)
    if res.degree == 0: 
        return None 
    else: 
        return res  


def _add_tokens_repetition(tok1: Token, tok2: Token): 
    res = Token(tok1.el, tok1.degree + tok2.degree)
    if res.degree % 2 == 0: 
        return None 
    else: 
        return res 


def reduce_pipe(seq: Iterable[Token], reduce_func): 
    stack = deque()
    for tok in seq: 
        if not stack: 
            stack.append(tok)
            continue
        
        prev = stack.pop()
        if prev.el == tok.el:
            res = reduce_func(prev, tok)
            if res is not None: 
                stack.append(res)
        else: 
            stack.append(prev)
            stack.append(tok)
    return stack 


def recover_word(seq: Iterable[Token], use_powering=True): 
    return ''.join(tok.get_repr(use_powering=use_powering) for tok in seq)


def free_reduce(word, alphabet: Iterable[str] = None, use_powering=False): 
    seq = tokenize(word, alphabet)
    seq = reduce_pipe(seq, _add_tokens_free)
    return recover_word(seq, use_powering=use_powering)


def reduce_repetitions(word, alphabet: Iterable[str] = None, use_powering=False):
    seq = tokenize(word, alphabet) 
    seq = reduce_pipe(seq, _add_tokens_repetition)
    return recover_word(seq, use_powering=use_powering)
