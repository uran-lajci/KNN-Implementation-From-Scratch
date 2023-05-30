import math
import pandas as pd


def calculate_euclidean_distance(p_coordinates, q_coordinates, num_dimensions):
    total_distance = 0
    for dimension in range(num_dimensions):
        difference_squared = pow((p_coordinates[dimension] - q_coordinates[dimension]), 2)
        total_distance += difference_squared
    euclidean_distance = math.sqrt(total_distance)
    return euclidean_distance


def calculate_neighbor_distances(train_dataset, test_dataset):
    neighbor_distances = {}
    num_dimensions = train_dataset.shape[1]
    for train_index, train_sample in train_dataset.iterrows():
        distance = calculate_euclidean_distance(train_sample, test_dataset, num_dimensions)
        neighbor_distances[train_index] = distance
    return neighbor_distances


def get_nearest_neighbors(distances_dict, k):
    sorted_distances = dict(sorted(distances_dict.items(), key=lambda item: item[1]))
    nearest_neighbors = {key: sorted_distances[key] for key in list(sorted_distances)[:k]}
    return nearest_neighbors


def calculate_regression_prediction(elements, k):
    prediction = sum(elements) / k
    return prediction


def calculate_mean_squared_error(actual_values, predicted_values):
    squared_errors = []
    for i in range(len(actual_values)):
        squared_error = (actual_values[i] - predicted_values[i]) ** 2
        squared_errors.append(squared_error)

    mean_squared_error = sum(squared_errors) / len(squared_errors)
    return mean_squared_error


if __name__ == "__main__":    
    df = pd.read_csv("dataset.csv")

    # train data
    train_features = df[["feature1", "feature2"]].iloc[0:9]
    train_targets = df[["target"]].iloc[0:9]

    # test data
    test_features = df[["feature1", "feature2"]].iloc[9]
    test_target = df[["target"]].iloc[9]

    distances = calculate_neighbor_distances(train_features, test_features)

    k = 3
    nearest_neighbors = get_nearest_neighbors(distances, k)

    selected_targets = []
    for neighbor_index in nearest_neighbors:
        selected_targets.append(train_targets.iloc[neighbor_index])
    
    knn_prediction = calculate_regression_prediction(selected_targets, k)
    print("Prediction with our code: ", float(knn_prediction))

    actual_value = float(test_target)
    predicted_value = float(knn_prediction)

    mse = calculate_mean_squared_error([actual_value], [predicted_value])
    formatted_mse = "{:.2f}".format(mse)
    print("Mean Squared Error:", formatted_mse)