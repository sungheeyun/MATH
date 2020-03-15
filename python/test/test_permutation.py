import unittest
from typing import Set

from itertools import permutations

from permutation import Permutation


class TestPermutation(unittest.TestCase):
    def test_basics(self) -> None:
        for perm in permutations((1, 2, 3)):
            permutation: Permutation = Permutation(perm)
            print(permutation)
            print(permutation.add_a_tail())
            self.assertEqual(permutation.add_a_tail().n_tuple, tuple(list(perm) + [4]))

        self.assertTrue(True)

    def test_permutation_swap(self) -> None:
        permutation4: Permutation = Permutation((1, 2, 3, 4))

        self.assertEqual(permutation4.swap(0, 0), Permutation((1, 2, 3, 4)))
        self.assertEqual(permutation4.swap(0, 1), Permutation((2, 1, 3, 4)))
        self.assertEqual(permutation4.swap(1, 0), Permutation((2, 1, 3, 4)))
        self.assertEqual(permutation4.swap(0, 2), Permutation((3, 2, 1, 4)))
        self.assertEqual(permutation4.swap(2, 0), Permutation((3, 2, 1, 4)))
        self.assertEqual(permutation4.swap(0, 3), Permutation((4, 2, 3, 1)))
        self.assertEqual(permutation4.swap(3, 0), Permutation((4, 2, 3, 1)))

        self.assertEqual(permutation4.swap(1, 1), Permutation((1, 2, 3, 4)))
        self.assertEqual(permutation4.swap(1, 2), Permutation((1, 3, 2, 4)))
        self.assertEqual(permutation4.swap(2, 1), Permutation((1, 3, 2, 4)))
        self.assertEqual(permutation4.swap(1, 3), Permutation((1, 4, 3, 2)))
        self.assertEqual(permutation4.swap(3, 1), Permutation((1, 4, 3, 2)))

        self.assertEqual(permutation4.swap(2, 2), Permutation((1, 2, 3, 4)))
        self.assertEqual(permutation4.swap(2, 3), Permutation((1, 2, 4, 3)))
        self.assertEqual(permutation4.swap(3, 2), Permutation((1, 2, 4, 3)))

        self.assertEqual(permutation4.swap(3, 3), Permutation((1, 2, 3, 4)))

        try:
            permutation4.swap(1, 4)
            self.assertTrue(False)
        except AssertionError:
            self.assertTrue(True)

    def test_permutation_set(self):
        even_set: Set[Permutation]
        odd_set: Set[Permutation]

        even_set, odd_set = Permutation.get_permutation_sets(4)

        print(sorted(even_set))
        print(sorted(odd_set))


if __name__ == '__main__':
    unittest.main()
