import numpy as np
from matplotlib import pyplot as plt

# EXERCISE 2 ===========================================================================================================
# Exercise 2 (b.) ------------------------------------------------------------------------------------------------------
train_cat = np.loadtxt('train_cat.txt', delimiter=',')
train_grass = np.loadtxt('train_grass.txt', delimiter=',')

n_train_cat = train_cat.shape[1]
n_train_grass = train_grass.shape[1]
prior_likelihood_cat = n_train_cat / (n_train_cat + n_train_grass)
prior_likelihood_grass = n_train_grass / (n_train_cat + n_train_grass)

mean_grass = (np.sum(train_grass, 1) / n_train_grass).reshape((-1, 1))
mean_cat = (np.sum(train_cat, 1) / n_train_cat).reshape((-1, 1))
d = mean_grass.shape[0]

cov_grass = (train_grass - mean_grass) @ (train_grass - mean_grass).T / (n_train_grass - 1)
cov_cat = (train_cat - mean_cat) @ (train_cat - mean_cat).T / (n_train_cat - 1)


# Exercise 2 (c.) ------------------------------------------------------------------------------------------------------
def make_max_function(_mu, _cov, _prior):
    _inverse_cov = np.linalg.inv(_cov)
    _det_cov = np.linalg.det(_cov)
    _log_det_cov = np.log(_det_cov)
    _log_prior = np.log(_prior)

    def _max_function(_x):
        _max_val = _log_prior - 0.5 * ((_x - _mu).swapaxes(-2, -1) @ _inverse_cov @ (_x - _mu) + _log_det_cov)
        return _max_val
    return _max_function


max_fun_grass = make_max_function(mean_grass, cov_grass, prior_likelihood_grass)
max_fun_cat = make_max_function(mean_cat, cov_cat, prior_likelihood_cat)

# cat_pixel_data = plt.imread('quiz_img_gray.jpg') / 255
cat_pixel_data = plt.imread('cat_grass.jpg') / 255
m, n = cat_pixel_data.shape
classification_mask = np.zeros(shape=(m - 8, n - 8), dtype=float)

# Generate Input Tensors (NumPy treats last two dimensions as the matrix -> mxnxNx1
input_tensor = np.empty(shape=(m - 8, n - 8, d, 1))
for i in range(m - 8):
    for j in range(n - 8):
        input_tensor[i, j, :, :] = cat_pixel_data[i:i + 8, j:j + 8].reshape((-1, 1))

max_vals_grass = max_fun_grass(input_tensor)[:, :, 0, 0]
max_vals_cat = max_fun_cat(input_tensor)[:, :, 0, 0]
classification_mask = max_vals_cat > max_vals_grass

fig_classification = plt.figure()
ax_classification = fig_classification.add_subplot(111)
ax_classification.imshow(1. - classification_mask, cmap='Greys', interpolation='nearest', vmin=0., vmax=1.)
ax_classification.set_xticks([])
ax_classification.set_yticks([])
ax_classification.set_title('Predicted Classification')
fig_classification.tight_layout()
fig_classification.savefig('hw3_p2c_output.png', format='png')

# Exercise 2 (d.) ------------------------------------------------------------------------------------------------------
truth_classification_image = plt.imread('truth.png')
mean_abs_error = np.sum(np.abs(classification_mask - truth_classification_image[:m - 8, :n - 8])) / ((m - 8) * (n - 8))


# Exercise 2 (e.) ------------------------------------------------------------------------------------------------------
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


cat_pixel_data_trial = plt.imread('cat_grass_new.jpg') / 255
cat_pixel_data_trial = rgb2gray(cat_pixel_data_trial)

# Plot Greyscale Image
fig_grey_cat_trial = plt.figure()
ax_grey_cat_trial = fig_grey_cat_trial.add_subplot(111)
ax_grey_cat_trial.imshow(1. - cat_pixel_data_trial, cmap='Greys', interpolation='nearest', vmin=0., vmax=1.)
ax_grey_cat_trial.set_xticks([])
ax_grey_cat_trial.set_yticks([])
fig_grey_cat_trial.tight_layout()
fig_grey_cat_trial.savefig('greyscale_cat_output.png', format='png')

m_trial, n_trial = cat_pixel_data_trial.shape
input_tensor_trial = np.empty(shape=(m_trial-8, n_trial-8, d, 1))
for i in range(m_trial - 8):
    for j in range(n_trial - 8):
        input_tensor_trial[i, j, :, :] = cat_pixel_data_trial[i:i + 8, j:j + 8].reshape((-1, 1))

max_vals_grass_trial = max_fun_grass(input_tensor_trial)[:, :, 0, 0]
max_vals_cat_trial = max_fun_cat(input_tensor_trial)[:, :, 0, 0]
classification_mask_trial = max_vals_cat_trial > max_vals_grass_trial

fig_classification = plt.figure()
ax_classification = fig_classification.add_subplot(111)
ax_classification.imshow(1. - classification_mask_trial, cmap='Greys', interpolation='nearest', vmin=0., vmax=1.)
ax_classification.set_xticks([])
ax_classification.set_yticks([])
ax_classification.set_title('Predicted Classification')
fig_classification.tight_layout()
fig_classification.savefig('hw3_p2e_output.png', format='png')

# EXERCISE 3 ===========================================================================================================
# Exercise 3 (b.) ------------------------------------------------------------------------------------------------------
truth_positive = np.asarray(truth_classification_image[:m - 8, :n - 8], dtype=bool)
truth_negative = np.logical_not(truth_positive)
n_truth_positive = np.sum(truth_positive)
n_truth_negative = np.sum(truth_negative)

# Bayesian Probability
prob_detect_bayes = np.sum(np.logical_and(classification_mask, truth_positive)) / n_truth_positive
prob_false_alarm_bayes = np.sum(np.logical_and(classification_mask, truth_negative)) / n_truth_negative

# Detection Probability for various values of tau
log_tau_max = np.max(max_vals_cat - max_vals_grass)
log_tau_min = np.min(max_vals_cat - max_vals_grass)
log_tau_vals = np.linspace(log_tau_min, log_tau_max, 100)
tau_vals = np.exp(log_tau_vals)
# tau_vals = np.logspace(-10, 5, 10)
# tau_vals = np.array((1.,), dtype=float)
# log_tau_vals = np.log(tau_vals)
prob_detect = np.empty(tau_vals.shape)
prob_false_alarm = np.empty(tau_vals.shape)

for idx, log_tau_val in enumerate(log_tau_vals):
    classification_positive = max_vals_cat > max_vals_grass + log_tau_val
    prob_detect[idx] = np.sum(np.logical_and(classification_positive, truth_positive)) / n_truth_positive
    prob_false_alarm[idx] = np.sum(np.logical_and(classification_positive, truth_negative)) / n_truth_negative

# Plot ROC Curve
fig_roc = plt.figure()
ax_roc = fig_roc.add_subplot(111)
ax_roc.grid()
ax_roc.plot(prob_false_alarm, prob_detect, linewidth=2)
ax_roc.plot(prob_false_alarm_bayes, prob_detect_bayes, 'r.', markersize=10)
ax_roc.set_xlabel(r'$P_F(\tau)$')
ax_roc.set_ylabel(r'$P_D(\tau)$')
ax_roc.set_title('ROC Curve')
fig_roc.tight_layout()
fig_roc.savefig('hw3_p3bc_output.png', format='png')
fig_roc.savefig('hw3_p3bc_output.svg', format='svg')

# Exercise 3 (d.) ------------------------------------------------------------------------------------------------------
design_matrix = np.vstack((train_cat.T, train_grass.T))
output = np.vstack((np.ones((n_train_cat, 1)), -np.ones((n_train_grass, 1))))
theta_hat = np.linalg.solve(design_matrix.T @ design_matrix, design_matrix.T @ output)

linear_prediction = (theta_hat.T @ input_tensor).squeeze()
tau_max_lin = np.max(linear_prediction)
tau_min_lin = np.min(linear_prediction)
tau_vals_lin = np.linspace(tau_min_lin, tau_max_lin, 100)

prob_detect_lin = np.empty(tau_vals_lin.shape)
prob_false_alarm_lin = np.empty(tau_vals_lin.shape)

for idx, tau_val in enumerate(tau_vals_lin):
    classification_positive = linear_prediction > tau_val
    prob_detect_lin[idx] = np.sum(np.logical_and(classification_positive, truth_positive)) / n_truth_positive
    prob_false_alarm_lin[idx] = np.sum(np.logical_and(classification_positive, truth_negative)) / n_truth_negative

# Plot ROC Curve
fig_roc = plt.figure()
ax_roc = fig_roc.add_subplot(111)
ax_roc.grid()
ax_roc.plot((0., 1.), (0., 1.), 'k', linewidth=2, label='Random Guess')
ax_roc.plot(prob_false_alarm_lin, prob_detect_lin, linewidth=2, label='Linear')
ax_roc.plot(prob_false_alarm, prob_detect, 'k--', linewidth=2, label='Gaussian')
ax_roc.plot(prob_false_alarm_bayes, prob_detect_bayes, 'r.', markersize=10, label='Bayesian')
ax_roc.set_xlabel(r'$P_F(\tau)$')
ax_roc.set_ylabel(r'$P_D(\tau)$')
ax_roc.set_title('ROC Curve (Linear)')
ax_roc.legend()
fig_roc.tight_layout()
fig_roc.savefig('hw3_p3d_output.png', format='png')
fig_roc.savefig('hw3_p3d_output.svg', format='svg')

plt.show()
