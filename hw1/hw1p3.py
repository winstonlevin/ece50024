import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

cols = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Exercise 3 (a) -------------------------------------------------------------------------------------------------------
x = np.linspace(-1, 1, 50)
l1 = sp.special.eval_legendre(1, x)
l2 = sp.special.eval_legendre(2, x)
l3 = sp.special.eval_legendre(3, x)
l4 = sp.special.eval_legendre(4, x)
l_poly = np.vstack((np.ones(l1.shape), l1, l2, l3, l4))
beta = np.array((-0.001, 0.01, 0.55, 1.5, 1.2)).reshape((-1, 1))
n_beta = beta.shape[0]

np.random.seed(3)
n_samples = 50
eps = np.random.normal(0., 0.2, n_samples)
y = (beta.T @ l_poly).flatten() + eps

# Scatter plot
fig_scatter = plt.figure()
ax = fig_scatter.add_subplot(111)
ax.scatter(x, y)
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
fig_scatter.tight_layout()
fig_scatter.savefig('p3a.svg', format='svg')

# Exercise 3 (c) -------------------------------------------------------------------------------------------------------
design_matrix = l_poly.T
output = y.reshape((-1, 1))
beta_hat = np.linalg.lstsq(design_matrix, output, rcond=None)[0]
y_hat = (beta_hat.T @ l_poly).flatten()

# Least-Squares scatter plot
fig_lstsq = plt.figure()
ax = fig_lstsq.add_subplot(111)
ax.scatter(x, y)
ax.plot(x, y_hat, color=cols[1])
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
fig_lstsq.tight_layout()
fig_lstsq.savefig('p3c.svg', format='svg')

# Exercise 3 (d) -------------------------------------------------------------------------------------------------------
output_outlier = output.copy()
output_outlier[(10, 16, 23, 37, 45), 0] = 5.
beta_hat_outlier = np.linalg.lstsq(design_matrix, output_outlier, rcond=None)[0]
y_hat_outlier = (beta_hat_outlier.T @ l_poly).flatten()

# Least-Squares scatter plot
fig_lstsq_outlier = plt.figure()
ax = fig_lstsq_outlier.add_subplot(111)
ax.scatter(x, output_outlier.flatten())
ax.plot(x, y_hat_outlier, color=cols[1])
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
fig_lstsq_outlier.tight_layout()
fig_lstsq_outlier.savefig('p3d.svg', format='svg')

# Exercise 3 (f) -------------------------------------------------------------------------------------------------------
c_linprog = np.vstack((np.zeros((n_beta, 1)), np.ones((n_samples, 1))))
eye_n_samples = np.eye(n_samples)
a_linprog = np.hstack((np.vstack((design_matrix, -design_matrix)), np.vstack((eye_n_samples, -eye_n_samples))))
b_linprog = np.vstack((output_outlier, -output_outlier))

a_eq_linprog = np.hstack((design_matrix, eye_n_samples))
b_eq_linprog = output_outlier

# sol = sp.optimize.linprog(c_linprog, a_linprog, b_linprog, bounds=(None, None))
sol = sp.optimize.linprog(c_linprog, A_eq=a_eq_linprog, b_eq=b_eq_linprog, bounds=(-50, 50))
beta_hat_linprog = sol.x[0:n_beta]
y_hat_linprog = (beta_hat_linprog.T @ l_poly).flatten()

# Plot
fig_linprog = plt.figure()
ax = fig_linprog.add_subplot(111)
ax.scatter(x, output_outlier.flatten())
ax.plot(x, y_hat_linprog, color=cols[1])
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
fig_linprog.tight_layout()
fig_linprog.savefig('p3d.svg', format='svg')

plt.show()
