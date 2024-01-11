import numpy as np
import scipy as sp
import matplotlib as mpl
from matplotlib import pyplot as plt

# Exercise 1 (a) -------------------------------------------------------------------------------------------------------
def gauss_pdf(_x, _mu: float = 0., _sig: float = 1.):
    _f = np.exp(-(_x - _mu)**2 / (2 * _sig**2)) / (2*np.pi*_sig**2)**0.5
    return _f

# Plot
x = np.linspace(-3., 3., 1000)
f = gauss_pdf(x)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, f)
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$f_X(x)$')
fig.tight_layout()
plt.savefig('p1a.svg', format='svg')

# Exercise 1 (b) -------------------------------------------------------------------------------------------------------
samples = np.random.normal(size=(1000,))
counts4, bins4, _ = plt.hist(samples, bins=4)
counts1k, bins1k, _ = plt.hist(samples, bins=4ZZ)

# Plot 1
plt.stairs(counts4, bins4)
