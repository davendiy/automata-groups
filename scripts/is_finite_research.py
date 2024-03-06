#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 17.01.2021
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi


from autogrp import *
import sys

H4 = AutomataGroup.generate_H4()
H4.disable_cache()


with open('logs/is_finite.log', 'w') as file:
    sys.stdout = file
    for el in all_words(H4.alphabet, max_len=7):
        el = H4(el)
        el.is_finite(verbose=True, algo=AS_WORDS)
        print()
