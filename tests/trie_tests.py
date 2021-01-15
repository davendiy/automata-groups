#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 15.01.2021
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

import unittest
from src.trie import Trie
import random
from string import ascii_lowercase, ascii_uppercase

SPACE = ascii_lowercase + ascii_uppercase
TESTS_AMOUNT = 100
MAX_LEN = 10000


class MyTestCase(unittest.TestCase):

    def test_add(self):
        all_words = set()
        test = Trie()
        for _ in range(TESTS_AMOUNT):
            length = random.randint(1, MAX_LEN)
            word = ''.join(random.choice(SPACE) for _ in range(length))
            all_words.add(word)
            test.add(word)
        for word in all_words:
            self.assertTrue(word in test)

    def test_iter(self):
        all_words = set()
        test = Trie()
        for _ in range(TESTS_AMOUNT):
            length = random.randint(1, MAX_LEN)
            word = ''.join(random.choice(SPACE) for _ in range(length))
            all_words.add(word)
            test.add(word)

        all_trie_words = {word for word in test}
        self.assertEqual(all_words, all_trie_words)

    def test_max_prefix(self):
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
            self.assertEqual((required_word, required_noise), test.max_prefix(string))


if __name__ == '__main__':
    unittest.main()
