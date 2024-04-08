#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 14.02.2020
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

from _autogrp_cython.permutation import Permutation
from _autogrp_cython.tools import (all_words, do_each, info, permute,
                                   random_el, reduce_repetitions as old_reduce)
from _autogrp_cython.reduction import free_reduce, reduce_repetitions
from _autogrp_cython.tokenizer import tokenize, as_tokens


from .automata import (AS_GROUP_ELEMENTS, AS_SHIFTED_WORDS, AS_WORDS,
                       ONLY_GENERAL, AutomataGroup, AutomataGroupElement,
                       DifferentGroupsError, MaximumOrderDeepError,
                       OutOfGroupError)

__all__ = ['reduce_repetitions', 'Permutation', 'all_words', 'permute',
           'AutomataGroup', 'AutomataGroupElement', 'AS_GROUP_ELEMENTS',
           'AS_SHIFTED_WORDS', 'AS_WORDS', 'ONLY_GENERAL', 'OutOfGroupError',
           'MaximumOrderDeepError', 'DifferentGroupsError', 'random_el',
           'do_each', 'info', 'tokenize', 'as_tokens', 'free_reduce']
