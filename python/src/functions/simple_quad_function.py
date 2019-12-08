from numpy import ndarray, linspace, pi, cos, sin, sqrt
from matplotlib.axes import Axes

from functions.function_base import FunctionBase


class SimpleQuadFunction(FunctionBase):
    """
    The function f: \reals^2 \to \reals

    f(x) = coef_1 * x_1^2 + coef_2 * x_2^2
    """

    num_points: int = 100

    def __init__(self, coef_1: float, coef_2: float) -> None:
        self.coef_1: float = coef_1
        self.coef_2: float = coef_2

    def draw_contour(self, axis: Axes, level: float, **kwargs) -> None:
        theta_array: ndarray = linspace(0.0, 2.0 * pi, SimpleQuadFunction.num_points)

        x_array: ndarray = sqrt(float(level) / self.coef_1) * cos(theta_array)
        y_array: ndarray = sqrt(float(level) / self.coef_2) * sin(theta_array)

        axis.plot(x_array, y_array, **kwargs)
