import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


def rayleigh_int(x):
    c = -np.cos(x)
    return 3/8 * (c + 1) + 1/8 * (c ** 3 + 1)

def rayleigh_diff(x):
    s = np.sin(x)
    c = np.cos(x)
    return 3/8 * s * (c^2 + 1)


if __name__ == '__main__':
    file_path = 'rayleigh angles.csv'
    start = 0
    end = np.pi
    sample_points = 10**3

    F = rayleigh_int
    f = rayleigh_diff

    points = np.arange(0,1+1/sample_points,1/sample_points)
    solutions = optimize.fsolve(lambda x: F(x) - points, points)

    data_points = np.vstack((points, solutions)).T
    np.savetxt(file_path, data_points, delimiter = ',')

    print(data_points)
    plt.plot(solutions)
    plt.show()

