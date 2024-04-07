#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 15.01.2021
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

from autogrp.trie import Trie, TriedDict
import random
from string import ascii_lowercase, ascii_uppercase

SPACE = ascii_lowercase + ascii_uppercase
TESTS_AMOUNT = 100
MAX_LEN = 1000



def test_trie_add():
    all_words = set()
    test = Trie()
    for _ in range(TESTS_AMOUNT):
        length = random.randint(1, MAX_LEN)
        word = ''.join(random.choice(SPACE) for _ in range(length))
        all_words.add(word)
        test.add(word)
    for word in all_words:
        assert word in test

def test_trie_iter():
    all_words = set()
    test = Trie()
    for _ in range(TESTS_AMOUNT):
        length = random.randint(1, MAX_LEN)
        word = ''.join(random.choice(SPACE) for _ in range(length))
        all_words.add(word)
        test.add(word)

    all_trie_words = {word for word in test}
    assert all_words == all_trie_words

def test_trie_max_prefix():
    all_words = set()
    test = Trie()
    for _ in range(TESTS_AMOUNT):
        length = random.randint(1, MAX_LEN)
        word = ''.join(random.choice(SPACE) for _ in range(length))
        all_words.add(word)
        test.add(word)

    words_space = list(all_words)
    sorted_words = sorted(words_space, key=len, reverse=True)
    for _ in range(TESTS_AMOUNT):
        required_noise = ''.join(random.choice(SPACE) for _ in range(MAX_LEN * 10))
        required_word = random.choice(words_space)
        string = required_word + required_noise
        for word in sorted_words:
            if string.startswith(word):
                required_word = word
                required_noise = string[len(word):]
                break
        assert (required_word, required_noise) == test.max_prefix(string)

def test_trie_max_prefix2():
    all_words = set()
    test = Trie()
    for _ in range(TESTS_AMOUNT):
        length = random.randint(1, MAX_LEN)
        word = ''.join(random.choice(SPACE) for _ in range(length))
        all_words.add(word)
        test.add(word)

    for el in all_words:
        prefix, left = test.max_prefix(el)
        assert (prefix, left) == (el, '')

def test_trie_remove():
    all_words = set()
    test = Trie()
    for _ in range(TESTS_AMOUNT):
        length = random.randint(1, MAX_LEN)
        word = ''.join(random.choice(SPACE) for _ in range(length))
        all_words.add(word)
        test.add(word)
    for _ in range(random.randint(1, len(all_words))):
        word = random.choice(list(all_words))
        all_words.remove(word)
        test.remove(word)
        assert not word in test

    res_words = set(test)
    assert all_words == res_words

def test_trie_len():
    all_words = set()
    test = Trie()
    for _ in range(TESTS_AMOUNT):
        length = random.randint(1, MAX_LEN)
        word = ''.join(random.choice(SPACE) for _ in range(length))
        all_words.add(word)
        test.add(word)

    assert len(all_words) == len(test)
    for _ in range(random.randint(1, len(all_words))):
        word = random.choice(list(all_words))
        all_words.remove(word)
        test.remove(word)
        assert not word in test

    assert len(all_words) == len(test)


def test_trieddict_set():
    all_words = {}
    test = TriedDict()
    for _ in range(TESTS_AMOUNT):
        length = random.randint(1, MAX_LEN)
        word = ''.join(random.choice(SPACE) for _ in range(length))
        value = random.random()
        all_words[word] = value
        test[word] = value

    for word in all_words:
        assert word in test

def test_trieddict_get():
    all_words = {}
    test = TriedDict()
    for _ in range(TESTS_AMOUNT):
        length = random.randint(1, MAX_LEN)
        word = ''.join(random.choice(SPACE) for _ in range(length))
        value = random.random()
        all_words[word] = value
        test[word] = value

    for word in all_words:
        assert all_words[word] == test[word]

def test_trieddict_del():
    all_words = {}
    test = TriedDict()
    for _ in range(TESTS_AMOUNT):
        length = random.randint(1, MAX_LEN)
        word = ''.join(random.choice(SPACE) for _ in range(length))
        value = random.random()
        all_words[word] = value
        test[word] = value
    for _ in range(random.randint(1, len(all_words))):
        word = random.choice(list(all_words))
        del all_words[word]
        del test[word]
        assert not word in test

    res_words = set(test)
    assert set(all_words) == res_words
