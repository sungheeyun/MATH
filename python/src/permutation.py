from __future__ import annotations
from typing import Tuple, Set, Union, List, Any


class Permutation:
    """
    Permutation of (1, 2, ...,. n)
    """

    def __init__(self, n_tuple: Union[list, tuple]) -> None:
        assert set(n_tuple) == set(range(1, len(n_tuple) + 1)), sorted(n_tuple)

        self.n_tuple: tuple = tuple(n_tuple)
        self.size: int = len(self.n_tuple)

    def add_a_tail(self) -> Permutation:
        """
        Add the number n + 1 to the permutation where n is the size of the original permutation.
        """
        return Permutation(list(self.n_tuple) + [len(self.n_tuple) + 1])

    def swap(self, idx1, idx2) -> Permutation:
        assert 0 <= idx1 < self.size, idx1
        assert 0 <= idx2 < self.size, idx2

        if idx1 == idx2:
            return self

        if idx1 > idx2:
            idx1, idx2 = idx2, idx1

        return Permutation(
            list(self.n_tuple[:idx1])
            + [self.n_tuple[idx2]]
            + list(self.n_tuple[idx1 + 1:idx2])
            + [self.n_tuple[idx1]]
            + list(self.n_tuple[idx2 + 1:])
        )

    def get_one_swap_permutation_set(self) -> Set[Permutation]:
        permutation_list: List[Permutation] = list()
        for idx1 in range(self.size - 1):
            for idx2 in range(idx1 + 1, self.size):
                assert idx1 < idx2
                permutation_list.append(self.swap(idx1, idx2))

        return set(permutation_list)

    def __repr__(self) -> str:
        return str(self.n_tuple)

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, Permutation)
        return self.n_tuple == other.n_tuple

    def __hash__(self) -> int:
        return hash(self.n_tuple)

    def __lt__(self, other: Permutation) -> bool:
        return self.n_tuple < other.n_tuple

    @staticmethod
    def get_permutation_sets(size: int) -> Tuple[Set[Permutation], Set[Permutation]]:
        assert size >= 2, size

        if size == 2:
            return set([Permutation((1, 2))]), set([Permutation((2, 1))])
        else:
            even_set_: Set[Permutation]
            odd_set_: Set[Permutation]
            even_set_, odd_set_ = Permutation.get_permutation_sets(size - 1)

            even_set_tailed: Set[Permutation] = set([permutation.add_a_tail() for permutation in even_set_])
            odd_set_tailed: Set[Permutation] = set([permutation.add_a_tail() for permutation in odd_set_])

            even_set: Set[Permutation] = set()
            odd_set: Set[Permutation] = set()

            for idx in range(size - 1):
                even_set.update([permutation.swap(idx, size - 1) for permutation in odd_set_tailed])
                odd_set.update([permutation.swap(idx, size - 1) for permutation in even_set_tailed])

            even_set.update(even_set_tailed)
            odd_set.update(odd_set_tailed)

            return even_set, odd_set
