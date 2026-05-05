from typing import Callable 


class FunctionDict:
    """
    A dictionary that computes its values on the fly using a provided function, and optionally caches the results.
    """
    def __init__(self, function: Callable, cache: bool = True):
        self.function = function
        self.cache = cache

        if cache:
            self._cache = dict()


    def __getitem__(self, key):
        if self.cache:
            if key not in self._cache:
                self._cache[key] = self.function(key)
            return self._cache[key]
        else:
            return self.function(key)

