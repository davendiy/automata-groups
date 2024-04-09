
from .tokenizer import as_tokens
from .trie import TriedDict


class TokenizedTriedDict(TriedDict):

    def __init__(self, alphabet=None):
        super().__init__()
        self._alphabet = alphabet

    def _to_tokens(self, word):
        if isinstance(word, str):
            return as_tokens(word, alphabet=self._alphabet, split=True)
        else:
            return word

    def __contains__(self, word) -> bool:
        return super().__contains__(self._to_tokens(word))

    def __setitem__(self, word, value):
        return super().__setitem__(self._to_tokens(word), value)

    def __getitem__(self, word):
        return super().__getitem__(self._to_tokens(word))

    def __delitem__(self, word):
        return super().__delitem__(self._to_tokens(word))

    # discuss: should max_prefix accept only tokens?
    def max_prefix(self, tokens):
        return super().max_prefix(tokens)
