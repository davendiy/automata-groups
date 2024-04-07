#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 16.01.2021
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

from autogrp import DiGraph
from autogrp.automata import AS_WORDS, AutomataGroup

# H3 = AutomataGroup.generate_H3()
# H4 = AutomataGroup.generate_H4()
# H3('aaaaaaaaabbbbcbbbbcbcbbbbbbbbbcbbbbaaaaaaabc').show()
# H3('a' * 30 + 'b' * 40 + 'c').show()
#
# H4('gcafbgca').is_finite()
#
# print(H4('gcafbgca').order())

# H4 = AutomataGroup.generate_H4()
# H4.one.disable_cache()
# test = DiGraph()
# H4('abcfc').order_graph(test, max_deep=4, loops=True)


H4 = AutomataGroup.generate_H4()
x = H4('abcfc')
# x.disable_cache()
# x.is_finite(verbose=True, algo=AS_SHIFTED_WORDS | ONLY_GENERAL)
#
# x.show()
#
# for el in x.dfs():
#     print(el)


H4('fdcgcaf').describe(algo=AS_WORDS)
