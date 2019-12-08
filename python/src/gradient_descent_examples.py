from numpy import ndarray, arange, power, linspace
from matplotlib import pyplot as plt

from functions.simple_quad_function import SimpleQuadFunction


if __name__ == "__main__":
    gamma: float = 10

    num_iterations: int = 20

    factor: float = (gamma - 1.0) / (gamma + 1.0)

    k_array: ndarray = arange(num_iterations)

    x_array_1: ndarray = gamma * power(factor, k_array)
    x_array_2: ndarray = power(-factor, k_array)

    simple_quad_function: SimpleQuadFunction = SimpleQuadFunction(0.5, 0.5 * gamma)

    contour_kwargs = dict(linestyle='--', color='k')
    fig, ax = plt.subplots()

    ax.plot(x_array_1, x_array_2, 'o-', color='k')
    for level in linspace(1, 50, 6):
        simple_quad_function.draw_contour(ax, level, **contour_kwargs)

    ax.axis('equal')

    fig.show()
