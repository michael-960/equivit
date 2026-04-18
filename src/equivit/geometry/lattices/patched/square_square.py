from ..square import Square

from .base import PatchedLattice


class SquareSquare(PatchedLattice):
    def __init__(self, N1: int, N2: int):
        self.square1 = Square(N1)
        self.square2 = Square(N2)

        self._setup_indices()
        self._setup_group_action()