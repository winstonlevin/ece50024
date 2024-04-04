import numpy as np

from model_classes import ImageDataset

categories_arr = np.loadtxt('category.csv', delimiter=',', skiprows=1, usecols=1, dtype=str)
n_categories, = categories_arr.shape

eval_data = np.loadtxt('compiled_test_results.csv', dtype=str, delimiter=',')
eval_accuracies = np.asarray(eval_data[0, :], dtype=float)
eval_predictions = eval_data[1:, :]
n_output, n_models = eval_predictions.shape


def ensemble_predictor(predictions, accuracies):
    n_output, n_models = predictions.shape

    # Determine how alike models are
    similarity_mat = np.empty(shape=(n_models, n_models))
    for idx in range(n_models):
        similarity_mat[idx, :] = np.sum(
            (predictions[:, idx:idx + 1] == predictions) & (predictions != ImageDataset.NO_IMAGE)
            , axis=0) / (predictions != ImageDataset.NO_IMAGE).sum(axis=0)

    # Use uniqueness of prediction to weight less similar models more heavily
    uniqueness_mat = np.linalg.inv(similarity_mat)

    # Convert predictions to integer and determine model similarities
    prediction_arr = np.zeros((n_output, n_models, n_categories), dtype=int)
    for idx in range(n_categories):
        prediction_arr[:, :, idx] = (predictions == categories_arr[idx]) * accuracies

    predictions_ensemble = (uniqueness_mat @ prediction_arr).sum(axis=1).argmax(axis=1)
    expected_accuracy = np.maximum(100., (uniqueness_mat @ accuracies.reshape((-1, 1))).sum())

    return predictions_ensemble, expected_accuracy


# # Run predictor on validation set and determine accuracy
# validation_data = np.loadtxt('compiled_validation_results.csv', dtype=str, delimiter=',')
# validation_accuracies = np.asarray(validation_data[0, :], dtype=float)
# validation_predictions = validation_data[1:, :]
#
# predictions_validation = ensemble_predictor(validation_predictions, validation_accuracies)
# targets_validation = np.loadtxt('data_validation.csv', delimiter=',', usecols=0, dtype=int)
#
# accuracy_ensemble = (predictions_validation == targets_validation).sum() / len(targets_validation)

# Run predictor on testing set
eval_data = np.loadtxt('compiled_test_results.csv', dtype=str, delimiter=',')
eval_accuracies = np.asarray(eval_data[0, :], dtype=float)
eval_predictions = eval_data[1:, :]

predictions_test, accuracy_ensemble = ensemble_predictor(eval_predictions, eval_accuracies)

# Save data in correct format ---------------------------------------------------------------------------------------- #
celebrity_names_arr = categories_arr[predictions_test]
file_name = 'test_results.csv'

try:
    previous_celebrity_names = np.loadtxt(file_name, dtype=str, skiprows=1, usecols=1, delimiter=',')
    num_changed = n_output - np.sum(celebrity_names_arr == previous_celebrity_names)
    print(f'Saving new results to "{file_name}" with expected accuracy {accuracy_ensemble:.2%}: '
          f'{num_changed}/{n_output} ({num_changed/n_output:.2%}) changed from previous results.')
except FileNotFoundError as _:
    print(f'Saving results to "{file_name}".')

image_id_arr = np.arange(0, len(eval_predictions[:, 0]), 1)
data_to_save = np.hstack((
    np.asarray(image_id_arr, dtype=str).reshape((-1, 1)),
    celebrity_names_arr.reshape((-1, 1))
))
np.savetxt(file_name, data_to_save, fmt='%s', delimiter=',', header='Id,Category', comments='')
