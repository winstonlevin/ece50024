import numpy as np
import sympy as sym
from matplotlib import pyplot as plt


# Exercise 2 (a) -------------------------------------------------------------------------------------------------------
def gauss_pdf(_x: np.ndarray, _mu: np.ndarray = np.zeros((1,)), _cov: np.ndarray = np.ones((1,))):
    _f = np.exp(
        -0.5 * np.moveaxis((_x - _mu), (_x.ndim-1,), (_x.ndim-2,)) @ np.linalg.solve(_cov, _x - _mu)
    ) / ((2 * np.pi) ** 2 * np.linalg.det(_cov)) ** 0.5
    _f = _f.squeeze()
    return _f


n_vals = 100
x1_vals = np.linspace(-1, 5, n_vals)
x2_vals = np.linspace(0, 10, n_vals)
x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)
mu = np.vstack((2, 6))
cov = np.vstack(((2, 1), (1, 2)))

x_grid = np.concatenate((x1_grid.reshape((n_vals, n_vals, 1, 1)), x2_grid.reshape((n_vals, n_vals, 1, 1))), axis=2)
f_grid = gauss_pdf(x_grid, mu, cov)
# f_grid = np.empty((n_vals, n_vals))
# ones_n = np.ones((n_vals, 1, 1))
# x1_vals_3d = x1_vals.reshape((-1, 1, 1))
# for idx, x1_val in enumerate(list(x1_vals)):
#     for jdx, x2_val in enumerate(list(x2_vals)):
#         x_arr = np.array((float(x1_val), float(x2_val))).reshape((2, 1))
#         f_grid[idx, jdx] = gauss_pdf(x_arr, mu, cov).flatten()[0]

fig_contour = plt.figure()
ax = fig_contour.add_subplot(111)
cont = ax.contour(x1_grid, x2_grid, f_grid)
ax.grid()
ax.axis('equal')
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
fig_contour.colorbar(cont)
fig_contour.tight_layout()
fig_contour.savefig('p2a_output.svg', format='svg')
fig_contour.savefig('p2a_output.png', format='png')

# Exercise 2 (b) -------------------------------------------------------------------------------------------------------
lam = sym.Matrix([[3, 0], [0, 1]])
sqrt2 = 1/sym.sqrt(2)
u = sym.Matrix([[sqrt2, -sqrt2], [sqrt2, sqrt2]])
cov_sym = u @ lam @ u.T
a_sym = u @ sym.sqrt(lam) @ u.T

# Exercise 2 (c) -------------------------------------------------------------------------------------------------------
np.random.seed(2)
mu_x = np.zeros((2,))
cov_x = np.eye(2)
n_samples = 5000
samples_x = np.random.multivariate_normal(mu_x, cov_x, n_samples).reshape((n_samples, 2, 1))

fig_scatter_x = plt.figure()
ax = fig_scatter_x.add_subplot(111)
ax.scatter(samples_x[:, 0, 0], samples_x[:, 1, 0], s=2)
ax.grid()
ax.axis('equal')
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
fig_scatter_x.colorbar(cont)
fig_scatter_x.tight_layout()
fig_scatter_x.savefig('p2c1_output.svg', format='svg')
fig_scatter_x.savefig('p2c1_output.png', format='png')

eig, eig_vec = np.linalg.eig(cov)
a = eig_vec @ np.diag(eig)**0.5 @ eig_vec.T
b = mu
samples_y = a @ samples_x + b

fig_scatter_y = plt.figure()
ax = fig_scatter_y.add_subplot(111)
ax.scatter(samples_y[:, 0, 0], samples_y[:, 1, 0], s=2)
cont = ax.contour(x1_grid, x2_grid, f_grid)
ax.grid()
ax.axis('equal')
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
fig_scatter_y.colorbar(cont)
fig_scatter_y.tight_layout()
fig_scatter_y.savefig('p2c2_output.svg', format='svg')
fig_scatter_y.savefig('p2c2_output.png', format='png')

plt.show()
