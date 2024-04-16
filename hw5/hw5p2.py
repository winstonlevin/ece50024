from typing import Optional
import math

import numpy as np
from matplotlib import pyplot as plt

# ==================================================================================================================== #
# PROBLEM 2                                                                                                            #
# ==================================================================================================================== #
# PROBLEM 2 a. ------------------------------------------------------------------------------------------------------- #
num_flips = 10  # 100,000 experiments of 1,000 coins flipped 10 times
num_coins = 1_000
num_trials = 100_000

m_heads = np.arange(0, num_flips+1, 1)
prob_m_heads = np.empty(shape=m_heads.shape, dtype=float)
for idx, m in enumerate(m_heads):
    prob_m_heads[idx] = \
        math.factorial(num_flips) / (math.factorial(num_flips - m) * math.factorial(m))
prob_m_heads *= 0.5 ** num_flips

prob_at_least_m_heads = np.zeros_like(prob_m_heads)
for idx, prob in enumerate(prob_m_heads):
    prob_at_least_m_heads[:idx+1] += prob

prob_at_least_m_heads_every_time = prob_at_least_m_heads ** num_coins

mu_min = (m_heads * prob_at_least_m_heads_every_time).sum() / num_flips

# PROBLEM 2 b. ------------------------------------------------------------------------------------------------------- #
rng = np.random.default_rng(seed=52)
print('Running 100,000 trials...')
coin_flips = rng.binomial(n=1, p=0.5, size=(num_trials, num_coins, num_flips))
print('Done :)')
freq_heads = coin_flips.sum(axis=-1)

idces_arr = np.kron(
    np.ones(shape=(num_trials, 1), dtype=int), np.arange(0, num_coins, 1).reshape((1, -1))
)

idx1 = np.zeros(shape=(num_trials, 1), dtype=int)
mask1 = idces_arr == idx1
idx_rand = rng.integers(low=0, high=num_coins, size=(num_trials, 1))
mask_rand = idces_arr == idx_rand
idx_min = freq_heads.argmin(axis=-1, keepdims=True)
mask_min = idces_arr == idx_min
idx_max = freq_heads.argmax(axis=-1, keepdims=True)
mask_max = idces_arr == idx_max


def pick_coin(_coin_flips, _mask):
    _coin = _coin_flips[_mask, ...]
    _v = _coin.sum(axis=-1) / num_flips
    return _coin, _v


coin1, v1 = pick_coin(coin_flips, mask1)
coin_rand, v_rand = pick_coin(coin_flips, mask_rand)
coin_min, v_min = pick_coin(coin_flips, mask_min)
coin_max, v_max = pick_coin(coin_flips, mask_max)


def plot_hist(_distro: np.ndarray, _x_lab: str = r'$V$', _y_lab: str = r'$P(V)$'):
    _counts, _bins = np.histogram(_distro)
    _dens = _counts / _distro.size
    _fig = plt.figure()
    _ax = _fig.add_subplot(111)
    _ax.grid()
    _ax.stairs(_dens, _bins, fill=True, label='Hist.')
    _ax.set_xlabel(_x_lab)
    _ax.set_ylabel(_y_lab)
    return _fig


for distro, x_lab, y_lab, output_name in zip(
        (v1, v_rand, v_min, v_max), (r'$V_1$', r'$V_{rand}$', r'$V_{min}$', r'$V_{max}$'),
        (r'$P(V_1)$', r'$P(V_{rand})$', r'$P(V_{min})$', r'$P(V_{max})$'),
        ('p2b_1_output', 'p2b_rand_output', 'p2b_min_output', 'p2b_quiz_output')
):
    fig = plot_hist(distro, _x_lab=x_lab, _y_lab=y_lab)
    fig.savefig(output_name + '.svg')
    fig.savefig(output_name + '.png')


# PROBLEM 2 c. ------------------------------------------------------------------------------------------------------- #
def calc_bounds(_distro, _num_data, _mu, _eps_vals, _x_label=r'$\epsilon$', _y_label=r'$V$'):
    _prob_est = np.empty_like(_eps_vals)
    _hoeff_bnd = np.empty_like(_eps_vals)
    for _idx, _eps in enumerate(_eps_vals):
        _prob_est[_idx] = (np.abs(_distro - _mu) > _eps).sum() / _distro.size
        _hoeff_bnd[_idx] = 2 * np.exp(-2 * _eps ** 2 * _num_data)

    _fig = plt.figure()
    _ax = _fig.add_subplot(111)
    _ax.grid()
    _ax.plot(_eps_vals, _prob_est, 'k', label='Estimate from Trials')
    _ax.plot(_eps_vals, _hoeff_bnd, color='0.5', label='Hoeffding Bound')
    _ax.legend()
    _ax.set_xlabel(_x_label)
    _ax.set_ylabel(_y_label)
    return _fig


eps_vals = np.arange(0., 0.5+0.05, 0.05)

for distro, mu, output_name, y_lab in zip(
        (v1, v_rand, v_min, v_min), (0.5, 0.5, mu_min, 0.5),
        ('p2c_1_output', 'p2c_rand_output', 'p2c_min_output', 'p2c_min_wrong_output'),
        (r'$P(|V_1 - \mu_1| > \epsilon)$', r'$P(|V_{rand} - \mu_{rand}| > \epsilon)$',
         r'$P(|V_{min} - \mu_{min}| > \epsilon)$', r'$P(|V_{min} - \mu_{min}| > \epsilon)$')
):
    fig = calc_bounds(distro, _num_data=num_flips, _mu=mu, _eps_vals=eps_vals, _y_label=y_lab)
    fig.savefig(output_name + '.svg')
    fig.savefig(output_name + '.png')

plt.show()
