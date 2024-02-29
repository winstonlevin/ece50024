import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt

# Load data
class0_data = np.loadtxt('homework4_class0.txt')
class1_data = np.loadtxt('homework4_class1.txt')
n_0 = class0_data.shape[0]
n_1 = class1_data.shape[0]

design_matrix = np.vstack((class0_data, class1_data))
design_matrix = np.hstack((design_matrix, np.ones((design_matrix.shape[0], 1))))
design_matrix_1 = design_matrix[n_0:, :]

# EXERCISE 2 ========================================================================================================= #
# EXERCISE 2 (b) ----------------------------------------------------------------------------------------------------- #
lam_ridge = 1e-4  # Constant to prevent theta -> infty

# Parameter Vector
theta_var_cvx = cp.Variable((3, 1), 'theta')

# Logistic Cost
n_data = class0_data.shape[0] + class1_data.shape[0]

# theta       = cvx.Variable((2,1))
loss = - cp.sum(design_matrix_1 @ theta_var_cvx) \
    + cp.sum(cp.log_sum_exp(cp.hstack([np.zeros((n_data, 1)), design_matrix @ theta_var_cvx]), axis=1))
reg = cp.sum_squares(theta_var_cvx)
prob = cp.Problem(cp.Minimize(loss/n_data + lam_ridge*reg))
prob.solve()

theta = theta_var_cvx.value

# EXERCISE 2 (c) ----------------------------------------------------------------------------------------------------- #
n_vals_contour = 100
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

# EXERCISE 2 (d) ----------------------------------------------------------------------------------------------------- #

plt.show()
