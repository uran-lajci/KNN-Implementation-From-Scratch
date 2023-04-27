import math
import operator
import warnings

warnings.filterwarnings("ignore")


def euclidean_distance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def get_neighbors(training_set, test_instance, k):
    distances = []
    length = len(test_instance) - 1
    for x in range(len(training_set)):
        dist = euclidean_distance(test_instance, training_set[x], length)
        distances.append((training_set[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def predict_regression(neighbors):
    prediction = sum(neighbor[-1] for neighbor in neighbors) / len(neighbors)
    return prediction


import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Load the dataset
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['MEDV'] = boston.target

# Split the dataset into training and testing sets
train_data, test_data, train_target, test_target = train_test_split(data[boston.feature_names], data['MEDV'], test_size=0.2, random_state=0)
train_data['MEDV'] = train_target
test_data['MEDV'] = test_target

# Make predictions on the test data
predictions = []
k = 5
for i in range(len(test_data)):
    neighbors = get_neighbors(train_data.values, test_data.values[i], k)
    prediction = predict_regression(neighbors)
    predictions.append(prediction)

# Compute the root-mean-square-error (RMSE)
rmse = math.sqrt(sum((test_target - predictions) ** 2) / len(test_target))
print(f"RMSE: {rmse}")
