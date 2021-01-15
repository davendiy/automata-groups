#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 15.01.2021
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

from collections import deque


class Trie:

    _end = '_end_'

    def __init__(self, *words):
        root = dict()
        self._root = root
        for word in words:
            self.add(word)

    def __contains__(self, word):
        cur_dict = self._root
        for letter in word:
            if letter not in cur_dict:
                return False
            cur_dict = cur_dict[letter]
        return self._end in cur_dict

    def add(self, word):
        cur_dict = self._root
        for letter in word:
            cur_dict = cur_dict.setdefault(letter, {})
        cur_dict[self._end] = self._end

    def max_prefix(self, string):
        cur_dict = self._root
        last_i = 0
        for i, letter in enumerate(string):
            if self._end in cur_dict:
                last_i = i
            if letter in cur_dict:
                cur_dict = cur_dict[letter]
            else:
                break
        return string[:last_i], string[last_i:]

    def remove(self, word):
        cur_dict = self._root
        for letter in word:
            if letter not in cur_dict:
                raise KeyError(f"There is no element '{word}' in given trie.")
            cur_dict = cur_dict[letter]
        if self._end not in cur_dict:
            raise KeyError(f"There is no element '{word}' in given trie.")
        else:
            del cur_dict[self._end]

    def _iter(self, root):
        for letter in root:
            if letter == self._end:
                yield ''
                continue
            for tail in self._iter(root[letter]):
                yield letter + tail

    def __str__(self):
        return f'Trie({self._root})'

    def __iter__(self):

        stack = deque()
        res_word = deque([''])

        stack.append((iter(self._root), self._root))
        while stack:
            cur_root_iter, cur_root = stack.pop()
            try:
                letter = next(cur_root_iter)
                if letter == self._end:
                    yield ''.join(res_word)
                    stack.append((cur_root_iter, cur_root))
                else:
                    stack.append((cur_root_iter, cur_root))
                    stack.append((iter(cur_root[letter]), cur_root[letter]))
                    res_word.append(letter)
            except StopIteration:
                res_word.pop()
