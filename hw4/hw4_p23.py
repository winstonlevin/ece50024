import numpy as np
import scipy as sp
import cvxpy as cp
from matplotlib import pyplot as plt

# Load data
class0_data = np.loadtxt('quiz4_class0.txt')
class1_data = np.loadtxt('quiz4_class1.txt')
# class0_data = np.loadtxt('homework4_class0.txt')
# class1_data = np.loadtxt('homework4_class1.txt')
n_0 = class0_data.shape[0]
n_1 = class1_data.shape[0]

design_matrix = np.vstack((class0_data, class1_data))
design_matrix = np.hstack((design_matrix, np.ones((design_matrix.shape[0], 1))))
design_matrix_0 = design_matrix[:n_0, :]
design_matrix_1 = design_matrix[n_0:, :]

# EXERCISE 2 ========================================================================================================= #
# EXERCISE 2 (b) ----------------------------------------------------------------------------------------------------- #
lam_ridge = 1e-2  # Constant to prevent theta -> infty
# lam_ridge = 1e-4  # Constant to prevent theta -> infty

# Parameter Vector
theta_var_cvx = cp.Variable((3, 1), 'theta')

# Logistic Cost
n_data = class0_data.shape[0] + class1_data.shape[0]

# theta       = cvx.Variable((2,1))
loss = - cp.sum(design_matrix_1 @ theta_var_cvx) \
    + cp.sum(cp.log_sum_exp(cp.hstack([np.zeros((n_data, 1)), design_matrix @ theta_var_cvx]), axis=1))
reg = cp.sum_squares(theta_var_cvx)
prob = cp.Problem(cp.Minimize(loss/n_data + lam_ridge*reg))
prob.solve(solver=cp.CLARABEL)

theta = theta_var_cvx.value

# EXERCISE 2 (c) ----------------------------------------------------------------------------------------------------- #
n_vals_contour = 500
x1_vals = np.linspace(-5., 10., n_vals_contour)
x2_vals = x1_vals.copy()
x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)
input_tensor = np.stack((x1_grid, x2_grid, np.ones(x1_grid.shape)), 2).reshape((n_vals_contour, n_vals_contour, -1, 1))
exp_vals = np.exp(-theta.T @ input_tensor).squeeze()
h_vals = 1 / (1 + exp_vals)

fig_discriminator = plt.figure()
ax_discriminator = fig_discriminator.add_subplot(111)
ax_discriminator.grid()
ax_discriminator.set_xlabel(r'$x_1$')
ax_discriminator.set_ylabel(r'$x_2$')
ax_discriminator.scatter(class0_data[:, 0], class0_data[:, 1], label='Class 0')
ax_discriminator.scatter(class1_data[:, 0], class1_data[:, 1], label='Class 1')
ax_discriminator.contour(x1_vals, x2_vals, h_vals, levels=(0., 0.5, 1.), cmap='Greys', linewidths=3)
ax_discriminator.legend()
ax_discriminator.set_title('Logistic Regression Discriminator')
fig_discriminator.tight_layout()
fig_discriminator.savefig('hw4_p2c_output.svg', format='svg')
fig_discriminator.savefig('hw4_p2c_output.png', format='png')

# EXERCISE 2 (d) ----------------------------------------------------------------------------------------------------- #
prior_0 = n_0 / (n_0 + n_1)
prior_1 = n_1 / (n_0 + n_1)

mean_0 = (np.sum(class0_data, 0) / n_0).reshape((-1, 1))
cov_0 = (class0_data.T - mean_0) @ (class0_data.T - mean_0).T / (n_0 - 1)

mean_1 = (np.sum(class1_data, 0) / n_1).reshape((-1, 1))
cov_1 = (class1_data.T - mean_1) @ (class1_data.T - mean_1).T / (n_1 - 1)


def make_max_function(_mu, _cov, _prior):
    _inverse_cov = np.linalg.inv(_cov)
    _det_cov = np.linalg.det(_cov)
    _log_det_cov = np.log(_det_cov)
    _log_prior = np.log(_prior)

    def _max_function(_x):
        _max_val = _log_prior - 0.5 * ((_x - _mu).swapaxes(-2, -1) @ _inverse_cov @ (_x - _mu) + _log_det_cov)
        return _max_val
    return _max_function


max_fun_0 = make_max_function(mean_0, cov_0, prior_0)
max_fun_1 = make_max_function(mean_1, cov_1, prior_1)

max_0_vals = max_fun_0(input_tensor[:, :, :2, :]).squeeze()
max_1_vals = max_fun_1(input_tensor[:, :, :2, :]).squeeze()
diff_vals = max_1_vals - max_0_vals
max_diff = np.max(np.abs(diff_vals))

fig_bayes = plt.figure()
ax_bayes = fig_bayes.add_subplot(111)
ax_bayes.grid()
ax_bayes.set_xlabel(r'$x_1$')
ax_bayes.set_ylabel(r'$x_2$')
ax_bayes.scatter(class0_data[:, 0], class0_data[:, 1], label='Class 0')
ax_bayes.scatter(class1_data[:, 0], class1_data[:, 1], label='Class 1')
ax_bayes.contour(
    x1_vals, x2_vals, max_1_vals - max_0_vals, levels=(-max_diff - 1., 0., max_diff + 1), cmap='Greys', linewidths=3
)
ax_bayes.legend()
ax_bayes.set_title('Bayesian Decision Boundary')
fig_bayes.tight_layout()
fig_bayes.savefig('hw4_p2d_output.svg', format='svg')
fig_bayes.savefig('hw4_p2d_output.png', format='png')


# EXERCISE 3 ========================================================================================================= #
# EXERCISE 3 (a) ----------------------------------------------------------------------------------------------------- #
h_kernal = 1.
design_matrix_2 = design_matrix[:, :2]

kernal_matrix = np.empty((n_data, n_data))
for idx in range(n_data):
    for jdx in range(n_data):
        kernal_matrix[idx, jdx] = np.exp(-np.sum((design_matrix_2[idx, :] - design_matrix_2[jdx, :])**2))

print('K[47:52,47:52:')
print(kernal_matrix[47:52, 47:52])

# EXERCISE 3 (c) ----------------------------------------------------------------------------------------------------- #
kernal_sqrt = sp.linalg.sqrtm(kernal_matrix)
kernal_1 = kernal_matrix[n_0:, :]
alpha_var_cvx = cp.Variable((n_data, 1), 'alpha')

# reg_alpha = cp.sum((kernal_sqrt @ alpha_var_cvx)**2)
reg_alpha = cp.quad_form(alpha_var_cvx, cp.psd_wrap(kernal_matrix))
loss_kernal = -(
                cp.sum(kernal_1 @ alpha_var_cvx)
                - cp.sum(cp.log_sum_exp(cp.hstack([np.zeros((n_data, 1)), kernal_matrix @ alpha_var_cvx]), axis=1))
              )

prob_alpha = cp.Problem(cp.Minimize(loss_kernal/n_data + lam_ridge*reg_alpha))
prob_alpha.solve(solver=cp.CLARABEL)

alpha = alpha_var_cvx.value

print('First two values of alpha:')
print(alpha[0:2])


# EXERCISE 3 (d) ----------------------------------------------------------------------------------------------------- #
def kernal_predictor(_input_tensor, _design_matrix, _alpha):
    _ndim = len(_input_tensor.shape)
    _x_data = np.expand_dims(_design_matrix, axis=tuple(np.append(np.arange(0, _ndim - 2, 1, dtype=int), -1)))
    _input_tensor = np.expand_dims(_input_tensor, axis=-3)
    _kern = np.exp(-np.sum((_input_tensor - _x_data)**2, axis=-2))
    _prediction = (_alpha.T @ _kern).squeeze()
    return _prediction


kern_prediction = kernal_predictor(input_tensor, design_matrix, alpha)
max_diff_kern = np.max(np.abs(kern_prediction))

fig_kern = plt.figure()
ax_kern = fig_kern.add_subplot(111)
ax_kern.grid()
ax_kern.set_xlabel(r'$x_1$')
ax_kern.set_ylabel(r'$x_2$')
ax_kern.scatter(class0_data[:, 0], class0_data[:, 1], label='Class 0')
ax_kern.scatter(class1_data[:, 0], class1_data[:, 1], label='Class 1')
ax_kern.contour(
    x1_vals, x2_vals, kern_prediction, levels=(-max_diff_kern - 1., 0.5, max_diff_kern + 1), cmap='Greys', linewidths=3
)
ax_kern.legend()
ax_kern.set_title('Kernal Decision Boundary')
fig_kern.tight_layout()
fig_kern.savefig('hw4_p3d_output.svg', format='svg')
fig_kern.savefig('hw4_p3d_output.png', format='png')

plt.show()
