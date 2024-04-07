#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 14.02.2020
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

from _autogrp_cython.permutation import Permutation
from _autogrp_cython.tools import (all_words, do_each, info, permute,
                                   random_el, reduce_repetitions)

from .automata import (AS_GROUP_ELEMENTS, AS_SHIFTED_WORDS, AS_WORDS,
                       ONLY_GENERAL, AutomataGroup, AutomataGroupElement,
                       DifferentGroupsError, MaximumOrderDeepError,
                       OutOfGroupError)

__all__ = ['reduce_repetitions', 'Permutation', 'all_words', 'permute',
           'AutomataGroup', 'AutomataGroupElement', 'AS_GROUP_ELEMENTS',
           'AS_SHIFTED_WORDS', 'AS_WORDS', 'ONLY_GENERAL', 'OutOfGroupError',
           'MaximumOrderDeepError', 'DifferentGroupsError', 'random_el',
           'do_each', 'info']
