from abc import ABC, abstractmethod

from matplotlib.axes import Axes


class FunctionBase(ABC):
    @abstractmethod
    def draw_contour(self, axis: Axes, level: float, **kwargs) -> None:
        """
        Draws a counter set.
        """
        pass
