#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 22.11.2019
# by David Zashkolny
# 3 course, comp math
# Taras Shevchenko National University of Kyiv
# email: davendiy@gmail.

from src.source import GroupElement

with open('critical.log') as file:
    interesting = list(set(file.read().split()))

for el in interesting:
    print(GroupElement(el))


tmp1 = GroupElement(interesting[0])
tmp2 = GroupElement(interesting[1])
print(tmp1, tmp2)

print(tmp1 * tmp2)
