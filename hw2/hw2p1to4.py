import numpy as np
from matplotlib import pyplot as plt
import cvxpy as cp
import csv

# Exercise 1 ===========================================================================================================
# Reading csv file for male data
with open("male_train_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    male_data_list = list(reader)
    male_data_types = male_data_list[:][0]
    male_data_arr = np.array(male_data_list[:][1:], dtype=float)
    male_bmi = male_data_arr[:, 1]
    male_stature_mm = male_data_arr[:, 2]
    male_bmi_normalized = male_bmi * 1e-1
    male_stature_normalized = male_stature_mm * 1e-3
csv_file.close()

# Reading csv file for female data
with open("female_train_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    female_data_list = list(reader)
    female_data_types = female_data_list[:][0]
    female_data_arr = np.array(female_data_list[:][1:], dtype=float)
    female_bmi = female_data_arr[:, 1]
    female_stature_mm = female_data_arr[:, 2]
    female_bmi_normalized = female_bmi * 1e-1
    female_stature_normalized = female_stature_mm * 1e-3
csv_file.close()

# Print data
print('| (idx) | Female BMI (norm.) | Female Stature (norm.) | Male BMI (norm.) | Male Stature (norm.) |')
print('|_______________________________________________________________________________________________|')

for idx, (f_bmi, f_stat, m_bmi, m_stat) in enumerate(zip(
        female_bmi_normalized, female_stature_normalized, male_bmi_normalized, male_stature_normalized
)):
    print(f'|   {idx:<3d} | {f_bmi:<18.4f} | {f_stat:<22.4f} | {m_bmi:<16.4f} | {m_stat:<20.4f} |')
    if idx == 9:
        break

# Exercise 2 ===========================================================================================================
# Exercise 2 (b) -------------------------------------------------------------------------------------------------------
n_male = male_bmi_normalized.size
n_female = female_bmi_normalized.size

# Assign +1 for male, -1 for female
output = np.vstack((
    np.ones((n_male, 1)), -np.ones((n_female, 1))
                    ))
bmis_normalized = np.concatenate((male_bmi_normalized, female_bmi_normalized))
stature_normalized = np.concatenate((male_stature_normalized, female_stature_normalized))
design_matrix = np.hstack((
    np.ones(output.shape), bmis_normalized.reshape((-1, 1)), stature_normalized.reshape((-1, 1))
))

theta_hat = np.linalg.solve(design_matrix.T @ design_matrix, design_matrix.T @ output)
n_theta = theta_hat.shape[0]

# Exercise 2 (c) -------------------------------------------------------------------------------------------------------
theta_hat_var_cvx = cp.Variable((n_theta, 1))
residual_cvx = output - design_matrix @ theta_hat_var_cvx
prob_cvx = cp.Problem(cp.Minimize(cp.sum_squares(residual_cvx)))
cost_cvx = prob_cvx.solve()
theta_hat_cvx = theta_hat_var_cvx.value


# Exercise 2 (e) -------------------------------------------------------------------------------------------------------
def objective(_theta):
    _residual = output - design_matrix @ _theta
    return (_residual.T @ _residual)[0, 0]


def line_step(_theta, _alpha):
    return _theta + 2 * _alpha * design_matrix.T @ (output - design_matrix @ _theta)


def exact_step_size(_theta):
    _grad = design_matrix.T @ (output - design_matrix @ _theta)
    _xx = design_matrix.T @ design_matrix
    return _grad.T @ _grad / (2 * _grad.T @ _xx @ _grad)


max_iter = 50_000
theta_hat_gd = 0. * theta_hat
training_loss_gd = np.empty((max_iter + 1,))
training_loss_gd[0] = objective(theta_hat_gd)

for idx_loss in range(1, max_iter+1):
    alpha = exact_step_size(theta_hat_gd)
    theta_hat_gd = line_step(theta_hat_gd, alpha)
    training_loss_gd[idx_loss] = objective(theta_hat_gd)

# Exercise 2 (f) -------------------------------------------------------------------------------------------------------
iterations = np.arange(0, max_iter+1, 1, dtype=int)
loss_min = objective(theta_hat)

fig_loss = plt.figure()
ax_loss = fig_loss.add_subplot(111)
ax_loss.grid()
ax_loss.set_xlabel('Iteration')
ax_loss.set_ylabel('A Posteriori Loss')
ax_loss.semilogx(iterations, training_loss_gd, linewidth=8)
ax_loss.semilogx(iterations[(0, -1), ], (loss_min, loss_min), 'k--', linewidth=8)
fig_loss.tight_layout()
fig_loss.savefig('hw2p2f.svg')
fig_loss.savefig('hw2p2f.png')


# Exercise 2 (g) -------------------------------------------------------------------------------------------------------
def line_step_momentum(_theta, _theta_prev, _alpha, _beta):
    _grad = design_matrix.T @ (output - design_matrix @ _theta)
    _grad_prev = design_matrix.T @ (output - design_matrix @ _theta_prev)
    return _theta + 2 * _alpha * ((1 - _beta) * _grad + _beta * _grad_prev), _theta


def exact_step_size_momentum(_theta, _theta_prev, _beta):
    _grad = design_matrix.T @ (output - design_matrix @ _theta)
    _grad_prev = design_matrix.T @ (output - design_matrix @ _theta_prev)
    _xx = design_matrix.T @ design_matrix
    _num = (1 - _beta) * _grad.T @ _grad + _beta * _grad_prev.T @ _grad
    _den = 2 * (
            (1 - _beta)**2 * _grad.T @ _xx @ _grad
            + 2 * _beta * (1 - _beta) * _grad.T @ _xx @ _grad_prev
            + _beta**2 * _grad_prev.T @ _xx @ _grad_prev
    )
    return _num / _den


theta_hat_mom = 0. * theta_hat
theta_hat_mom_prev = 0. * theta_hat
training_loss_mom = np.empty((max_iter + 1,))
training_loss_mom[0] = objective(theta_hat_gd)
beta = 0.9

for idx_loss in range(1, max_iter+1):
    alpha = exact_step_size_momentum(theta_hat_mom, theta_hat_mom_prev, beta)
    theta_hat_mom, theta_hat_mom_prev = line_step_momentum(theta_hat_mom, theta_hat_mom_prev, alpha, beta)
    training_loss_mom[idx_loss] = objective(theta_hat_mom)

# Exercise 2 (h) -------------------------------------------------------------------------------------------------------
fig_loss_mom = plt.figure()
ax_loss_mom = fig_loss_mom.add_subplot(111)
ax_loss_mom.grid()
ax_loss_mom.set_xlabel('Iteration')
ax_loss_mom.set_ylabel('A Posteriori Loss')
ax_loss_mom.semilogx(iterations, training_loss_mom, linewidth=8)
ax_loss_mom.semilogx(iterations[(0, -1), ], (loss_min, loss_min), 'k--', linewidth=8)
fig_loss_mom.tight_layout()
fig_loss_mom.savefig('hw2p2h.svg')
fig_loss_mom.savefig('hw2p2h.png')


# Exercise 3 ===========================================================================================================
# Exercise 3 (a) -------------------------------------------------------------------------------------------------------
def stature_boundary(_bmi, _theta):
    return -(_theta[1] * _bmi + _theta[0])/_theta[2]


n_est = 1_000
bmi_vals = np.linspace(np.min(bmis_normalized), np.max(bmis_normalized), n_est)
stature_boundary_vals = stature_boundary(bmi_vals, theta_hat)
true_male = np.nonzero(output > 0)[0]
true_female = np.nonzero(output < 0)[0]

fig_classification = plt.figure()
ax_classification = fig_classification.add_subplot(111)
ax_classification.grid()
ax_classification.set_xlabel('BMI (norm.)')
ax_classification.set_ylabel('Stature (norm.)')
ax_classification.scatter(bmis_normalized[true_male], stature_normalized[true_male], marker='o', color='b', facecolor=None)
ax_classification.scatter(bmis_normalized[true_female], stature_normalized[true_female], marker='.', color='r', facecolor=None)
ax_classification.plot(bmi_vals, stature_boundary_vals, 'k', linewidth=2)
fig_classification.tight_layout()
fig_classification.savefig('hw2p3a.svg')
fig_classification.savefig('hw2p3a.png')

# Exercise 3 (b) -------------------------------------------------------------------------------------------------------
with open("male_test_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    male_test_data_list = list(reader)
    male_test_data_types = male_test_data_list[:][0]
    male_test_data_arr = np.array(male_test_data_list[:][1:], dtype=float)
    male_test_bmi = male_test_data_arr[:, 1]
    male_test_stature_mm = male_test_data_arr[:, 2]
    male_test_bmi_normalized = male_test_bmi * 1e-1
    male_test_stature_normalized = male_test_stature_mm * 1e-3
csv_file.close()

# Reading csv file for female data
with open("female_test_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    female_test_data_list = list(reader)
    female_test_data_types = female_test_data_list[:][0]
    female_test_data_arr = np.array(female_test_data_list[:][1:], dtype=float)
    female_test_bmi = female_test_data_arr[:, 1]
    female_test_stature_mm = female_test_data_arr[:, 2]
    female_test_bmi_normalized = female_test_bmi * 1e-1
    female_test_stature_normalized = female_test_stature_mm * 1e-3
csv_file.close()

n_male_test = np.shape(male_test_bmi_normalized)[0]
n_female_test = np.shape(female_test_bmi_normalized)[0]

test_matrix_female = np.hstack((
    np.ones((n_female_test, 1)), female_test_bmi_normalized.reshape((-1, 1)), female_test_stature_normalized.reshape((-1, 1))
))
classification_test_female = np.sign(test_matrix_female @ theta_hat)
false_positive = np.sum(classification_test_female > 0)
type_1_error = false_positive / n_female_test

test_matrix_male = np.hstack((
    np.ones((n_male_test, 1)), male_test_bmi_normalized.reshape((-1, 1)), male_test_stature_normalized.reshape((-1, 1))
))
classification_test_male = np.sign(test_matrix_male @ theta_hat)
false_negative = np.sum(classification_test_male < 0)
type_2_error = false_negative / n_male_test

n_identified_males_correct = n_male_test - false_negative
n_identified_males = n_identified_males_correct + false_positive

precision = n_identified_males_correct / n_identified_males
recall = n_identified_males_correct / n_male_test

fig_classification_test = plt.figure()
ax_classification_test = fig_classification_test.add_subplot(111)
ax_classification_test.grid()
ax_classification_test.set_xlabel('BMI (norm.)')
ax_classification_test.set_ylabel('Stature (norm.)')
ax_classification_test.scatter(male_test_bmi_normalized, male_test_stature_normalized, marker='o', color='b', facecolor=None)
ax_classification_test.scatter(female_test_bmi_normalized, female_test_stature_normalized, marker='.', color='r', facecolor=None)
ax_classification_test.plot(bmi_vals, stature_boundary_vals, 'k', linewidth=2)
fig_classification_test.tight_layout()
fig_classification_test.savefig('hw2p3b.svg')
fig_classification_test.savefig('hw2p3b.png')

# Exercise 4 ===========================================================================================================
# Exercise 4 (a) -------------------------------------------------------------------------------------------------------
lam_d = np.arange(0.1, 10, 0.1)

tha_lam = np.empty((n_theta, lam_d.size))
norm_res = np.empty(lam_d.shape)
norm_tha = np.empty(lam_d.shape)

XTX = design_matrix.T @ design_matrix
XTy = design_matrix.T @ output

for idx, lam_i in enumerate(lam_d):
    tha_lam[:, idx] = np.linalg.solve(XTX + lam_i * np.eye(n_theta), XTy).flatten()
    norm_res[idx] = np.linalg.norm(design_matrix @ tha_lam[:, idx] - output)**2
    norm_tha[idx] = np.linalg.norm(tha_lam[:, idx])**2

norm_res_lab = r'$\| X \hat{\theta}_{\lambda} - y \|^2$'
norm_tha_lab = r'$\| \hat{\theta}_{\lambda} \|^2$'
lam_lab = r'$\lambda$'
ydata = (norm_res, norm_res, norm_tha)
xdata = (norm_tha, lam_d, lam_d)
ylabs = (norm_res_lab, norm_res_lab, norm_tha_lab)
xlabs = (norm_tha_lab, lam_lab, lam_lab)
n_y = len(ylabs)

fig_ridge = plt.figure()
axes_ridge = []

for idx, y in enumerate(ydata):
    axes_ridge.append(fig_ridge.add_subplot(n_y, 1, idx+1))
    ax = axes_ridge[-1]
    ax.grid()
    ax.set_xlabel(xlabs[idx])
    ax.set_ylabel(ylabs[idx])
    ax.plot(xdata[idx], y, linewidth=2)

fig_ridge.tight_layout()
fig_ridge.savefig('hw2p4a.svg')
fig_ridge.savefig('hw2p4a.png')

plt.show()
