import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

# Exercise 3 (a) -------------------------------------------------------------------------------------------------------
x = np.linspace(-1, 1, 50)
l1 = sp.special.eval_legendre(1, x)
l2 = sp.special.eval_legendre(2, x)
l3 = sp.special.eval_legendre(3, x)
l4 = sp.special.eval_legendre(4, x)
l_poly = np.vstack((np.ones(l1.shape), l1, l2, l3, l4))
beta = np.array((-0.001, 0.01, 0.55, 1.5, 1.2)).reshape((-1, 1))

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

# Exercise 3 (b) -------------------------------------------------------------------------------------------------------


plt.show()
