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

fig_fx = plt.figure()
ax = fig_fx.add_subplot(111)
ax.plot(x, f)
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$f_X(x)$')
fig_fx.tight_layout()
plt.savefig('p1a.svg', format='svg')

# Exercise 1 (b) -------------------------------------------------------------------------------------------------------
np.random.seed(1)
samples = np.random.normal(size=(1000,))
counts4, bins4, _ = plt.hist(samples, bins=4, density=True)
counts1k, bins1k, _ = plt.hist(samples, bins=1000, density=True)

# Estimated mean/std
mean_est, std_est = sp.stats.norm.fit(samples)
print('Based on the 1,000 samples the estimated Mean/STD are:')
print((mean_est, std_est))

x_est = np.linspace(np.min(samples-1), np.max(samples+1), 1000)
f_est = gauss_pdf(x_est, _mu=mean_est, _sig=std_est)

# Plot 1
fig_hist4 = plt.figure()
ax = fig_hist4.add_subplot(111)
ax.stairs(counts4, bins4, fill=True)
ax.plot(x_est, f_est)
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'Dens.')
fig_hist4.tight_layout()
plt.savefig('p1b1.svg', format='svg')

# Plot 2
fig_hist1k = plt.figure()
ax = fig_hist1k.add_subplot(111)
ax.stairs(counts1k, bins1k, fill=True)
ax.plot(x_est, f_est)
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'Dens.')
fig_hist1k.tight_layout()
plt.savefig('p1b2.svg', format='svg')

# Exercise 1 (c) -------------------------------------------------------------------------------------------------------


# Display all plots ----------------------------------------------------------------------------------------------------
plt.show()
