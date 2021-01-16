#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 16.01.2021
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

from src.source3 import *


H3 = AutomataGroup.generate_H3()
H4 = AutomataGroup.generate_H4()
H3('aaaaaaaaabbbbcbbbbcbcbbbbbbbbbcbbbbaaaaaaabc').show()
H3('a' * 30 + 'b' * 40 + 'c').show()

H4('gcafbgca')._is_finite2(use_cache=False)

print(H4('gcafbgca').order())
