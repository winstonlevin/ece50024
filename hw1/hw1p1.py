import numpy as np
import scipy as sp
import matplotlib as mpl
from matplotlib import pyplot as plt


# Exercise 1 (a) -------------------------------------------------------------------------------------------------------
def gauss_pdf(_x, _mu: float = 0., _sig: float = 1.):
    _f = np.exp(-(_x - _mu)**2 / (2 * _sig**2)) / (2*np.pi*_sig**2)**0.5
    return _f


x = np.linspace(-3., 3., 1000)
f = gauss_pdf(x)

# Plot
fig_fx = plt.figure()
ax = fig_fx.add_subplot(111)
ax.plot(x, f)
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$f_X(x)$')
fig_fx.tight_layout()
fig_fx.savefig('p1a.svg', format='svg')

# Exercise 1 (b) -------------------------------------------------------------------------------------------------------
np.random.seed(1)
samples = np.random.normal(size=(1000,))
dens4, bins4, _ = plt.hist(samples, bins=4, density=True)
dens1k, bins1k, _ = plt.hist(samples, bins=1000, density=True)

# Estimated mean/std
mean_est, std_est = sp.stats.norm.fit(samples)
print('Based on the 1,000 samples the estimated Mean/STD are:')
print((mean_est, std_est))

x_est = np.linspace(np.min(samples-1), np.max(samples+1), 1000)
f_est = gauss_pdf(x_est, _mu=mean_est, _sig=std_est)

# Plot 1
fig_hist4 = plt.figure()
ax = fig_hist4.add_subplot(111)
ax.stairs(dens4, bins4, fill=True)
ax.plot(x_est, f_est)
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'Dens.')
fig_hist4.tight_layout()
fig_hist4.savefig('p1b1.svg', format='svg')

# Plot 2
fig_hist1k = plt.figure()
ax = fig_hist1k.add_subplot(111)
ax.stairs(dens1k, bins1k, fill=True)
ax.plot(x_est, f_est)
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'Dens.')
fig_hist1k.tight_layout()
fig_hist1k.savefig('p1b2.svg', format='svg')


# Exercise 1 (c) -------------------------------------------------------------------------------------------------------
def cross_validation_score(_n_samples, _h, _dens):
    _score = (2. - (_n_samples+1) * np.dot(_dens, _dens)) / ((_n_samples-1) * _h)
    return _score


n_bins = np.arange(start=1, stop=200+1, step=1, dtype=int)
scores = np.empty(n_bins.shape)
data_range = np.max(samples) - np.min(samples)
n_samples = samples.shape[0]

for idx, m in enumerate(n_bins):
    h = data_range / m
    dens, _, __ = plt.hist(samples, bins=m)
    dens = dens / n_samples
    scores[idx] = cross_validation_score(_n_samples=n_samples, _h=h, _dens=dens)

# Plot 1
fig_scores = plt.figure()
ax = fig_scores.add_subplot(111)
ax.plot(n_bins, scores, linewidth=2)
ax.grid()
ax.set_xlabel('No. Bins')
ax.set_ylabel('Cross-Validation Score')
fig_scores.tight_layout()
fig_scores.savefig('p1c1.svg', format='svg')

m_opt = np.argmin(scores)
print('The number of bins with the lowest cost is:')
print(m_opt)

# Plot 2
fig_histopt = plt.figure()
ax = fig_histopt.add_subplot(111)
ax.hist(samples, bins=m_opt, density=True)
ax.plot(x_est, f_est)
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'Dens.')
fig_histopt.tight_layout()
fig_histopt.savefig('p1c2.svg', format='svg')

# Display all plots ----------------------------------------------------------------------------------------------------
plt.show()
