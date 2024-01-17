import numpy as np
from matplotlib import pyplot as plt


# Exercise 2 (a) -------------------------------------------------------------------------------------------------------
def gauss_pdf(_x: np.ndarray, _mu: np.ndarray = np.zeros((1,)), _cov: np.ndarray = np.ones((1,))):
    _f = np.exp(
        -0.5 * np.moveaxis((_x - _mu), (_x.ndim-1,), (_x.ndim-2,)) @ np.linalg.solve(_cov, _x - _mu)
    ) / ((2 * np.pi) ** 2 * np.linalg.det(_cov)) ** 0.5
    return _f


n_vals = 100
x1_vals = np.linspace(-1, 5, n_vals)
x2_vals = np.linspace(0, 10, n_vals)
mu = np.vstack((2, 6))
cov = np.vstack(((2, 1), (1, 2)))

f_grid = np.empty((n_vals, n_vals))
ones_n = np.ones((n_vals, 1, 1))
x1_vals_3d = x1_vals.reshape((-1, 1, 1))
for idx, x1_val in enumerate(list(x1_vals)):
    for jdx, x2_val in enumerate(list(x2_vals)):
        x_arr = np.array((float(x1_val), float(x2_val))).reshape((2, 1))
        f_grid[idx, jdx] = float(gauss_pdf(x_arr, mu, cov))

fig_contour = plt.figure()
ax = fig_contour.add_subplot(111)
ax.contour(x1_vals, x2_vals, f_grid)
ax.grid()
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
plt.show()
