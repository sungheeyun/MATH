from typing import Set, List

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from freq_used.plotting import get_figure

from permutation import Permutation


if __name__ == "__main__":
    size: int = 4
    text_space: float = 0.1

    even_set: Set[Permutation]
    odd_set: Set[Permutation]

    even_set, odd_set = Permutation.get_permutation_sets(size)

    assert len(even_set) == len(odd_set)

    even_perm_list: List[Permutation] = sorted(even_set)
    odd_perm_list: List[Permutation] = sorted(odd_set)

    figure: Figure = get_figure(1, 1, left_margin=3, right_margin=3, axis_width=3, axis_height=6)
    axis: Axes = figure.get_axes()[0]

    for idx, even_permutation in enumerate(even_perm_list):
        axis.text(-text_space, -idx, str(even_permutation), ha='right', va='center')
        axis.text(1.0 + text_space, -idx, str(odd_perm_list[idx]), ha='left', va='center')

        one_swap_permutation_set: Set[Permutation] = even_permutation.get_one_swap_permutation_set()

        for odd_idx, odd_permutation in enumerate(odd_perm_list):
            if odd_permutation in one_swap_permutation_set:
                axis.plot((0, 1), (-idx, -odd_idx), "k-")

    axis.set_xlim((0, 1))
    axis.set_ylim((-len(even_perm_list), 1))
    axis.axis("off")

    figure.show()
