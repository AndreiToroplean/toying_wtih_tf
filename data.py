import numpy as np
import tensorflow as tf


def f(x):
    return 1 / x


def gen_x_y(func=f, n_points=1000, bounds=(-10, 10), seed=0):
    np.random.seed(seed)

    x = np.random.uniform(*bounds, n_points)
    y = func(x)
    return x, y


x, y = gen_x_y()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(x, y, ".")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.show()
