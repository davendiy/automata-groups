#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 19.01.2021
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

from autogrp import *
import sys

H4 = AutomataGroup.generate_H4()

stdout = sys.stdout

with open('logs/finite_els.log', 'w') as file:
    sys.stdout = file

    for el in all_words(H4.alphabet, max_len=5):
        stdout.write('\b' * 1000 + el)
        el = H4(el)
        if el.is_finite():
            el.describe(show_structure=False)
