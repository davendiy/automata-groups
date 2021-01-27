#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 14.02.2020
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

from .permutation import Permutation
from .tools import reduce_repetitions, all_words, permute, random_el
from .automata import AutomataGroup, AutomataGroupElement, \
        AS_GROUP_ELEMENTS, AS_WORDS, AS_SHIFTED_WORDS, ONLY_GENERAL,\
        DifferentGroupsError, OutOfGroupError, MaximumOrderDeepError

__all__ = ['reduce_repetitions', 'Permutation', 'all_words', 'permute',
           'AutomataGroup', 'AutomataGroupElement', 'AS_GROUP_ELEMENTS',
           'AS_SHIFTED_WORDS', 'AS_WORDS', 'ONLY_GENERAL', 'OutOfGroupError',
           'MaximumOrderDeepError', 'DifferentGroupsError', 'random_el']
