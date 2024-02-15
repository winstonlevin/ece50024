import numpy as np
from matplotlib import pyplot as plt

# Exercise 2 (b.) ------------------------------------------------------------------------------------------------------
train_cat = np.matrix(np.loadtxt('train_cat.txt', delimiter=','))
train_grass = np.matrix(np.loadtxt('train_grass.txt', delimiter=','))

n_train_cat = train_cat.shape[1]
n_train_grass = train_grass.shape[1]
prior_likelihood_cat = n_train_cat / (n_train_cat + n_train_grass)
prior_likelihood_grass = n_train_grass / (n_train_cat + n_train_grass)

mean_grass = np.sum(train_grass, 1) / n_train_grass
mean_cat = np.sum(train_cat, 1) / n_train_cat

cov_grass = (train_grass - mean_grass) @ (train_grass - mean_grass).T / (n_train_grass - 1)
cov_cat = (train_cat - mean_cat) @ (train_cat - mean_cat).T / (n_train_cat - 1)


# Exercise 2 (c.) ------------------------------------------------------------------------------------------------------
def make_max_function(_mu, _cov, _prior):
    _inverse_cov = np.linalg.inv(_cov)
    _det_cov = np.linalg.det(_cov)
    _log_det_cov = np.log(_det_cov)
    _log_prior = np.log(_prior)

    def _max_function(_x):
        _max_val = 0.5 * ((_x - _mu).T @ _inverse_cov @ (_x - _mu) - _log_det_cov) + _log_prior
        return _max_val
    return _max_function


max_fun_grass = make_max_function(mean_grass, cov_grass, prior_likelihood_grass)
max_fun_cat = make_max_function(mean_cat, cov_cat, prior_likelihood_cat)

cat_pixel_data = plt.imread('cat_grass.jpg') / 255
m, n = cat_pixel_data.shape
classification_image = np.zeros(shape=(m-8, n-8), dtype=float)

for i in range(m - 8):
    for j in range(n - 8):
        cat_pixel_block = cat_pixel_data[i:i + 8, j:j + 8]
        input_vector = cat_pixel_block.reshape((-1, 1))
        max_val_grass = max_fun_grass(input_vector)[0, 0]
        max_val_cat = max_fun_cat(input_vector)[0, 0]

        if max_val_cat > max_val_grass:
            classification_image[i, j] = 1.

fig_classification = plt.figure()
ax_classification = fig_classification.add_subplot(111)
ax_classification.imshow(classification_image, cmap='Greys', interpolation='nearest', vmin=0., vmax=1.)
ax_classification.set_xticks([])
ax_classification.set_yticks([])
ax_classification.set_title('Predicted Classification')
