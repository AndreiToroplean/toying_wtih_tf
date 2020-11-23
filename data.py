import numpy as np


def f(x):
    return 1 / x


np.random.seed(0)

bounds = (-10, 10)
n_points = 1000

x = np.random.uniform(*bounds, n_points)
y = f(x)
print(x, y, sep="\n")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.plot(x, y, ".")
    plt.xlim(*bounds)
    plt.ylim(-10, 10)
    plt.show()
