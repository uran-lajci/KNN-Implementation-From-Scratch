import math
import operator
import warnings

import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")


def euclidean_distance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def get_neighbors(training_set, test_instance, k, distance_function):
    distances = []
    length = len(test_instance) - 1
    for x in range(len(training_set)):
        dist = distance_function(test_instance, training_set[x], length)
        distances.append((training_set[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def predict_regression(neighbors):
    prediction = sum(neighbor[-1] for neighbor in neighbors) / len(neighbors)
    return prediction


# Load the dataset
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['MEDV'] = boston.target

# Split the dataset into training and testing sets
train_data, test_data, train_target, test_target = train_test_split(data[boston.feature_names], data['MEDV'], test_size=0.2, random_state=0)
train_data['MEDV'] = train_target
test_data['MEDV'] = test_target

# Test with different k values
k_values = [3, 5, 7]

for k in k_values:
    predictions = []
    for i in range(len(test_data)):
        neighbors = get_neighbors(train_data.values, test_data.values[i], k, euclidean_distance)
        prediction = predict_regression(neighbors)
        predictions.append(prediction)

    # Compute the performance measures
    rmse = math.sqrt(mean_squared_error(test_target, predictions))
    mae = mean_absolute_error(test_target, predictions)
    r2 = r2_score(test_target, predictions)

    print(f"Distance: {euclidean_distance.__name__}, k: {k}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2 Score: {r2}\n")

######################################################################

    # import pandas as pd
    # from sklearn.neighbors import KNeighborsRegressor

    # # Separate the features (X) and target (y)
    # X = df[['feature1', 'feature2']]
    # y = df['target']

    # # Initialize and fit the KNeighborsRegressor with k=3
    # knn = KNeighborsRegressor(n_neighbors=3)
    # knn.fit(X, y)

    # # Extract the last row from the dataset for prediction
    # new_data = df.iloc[[-1], :-1]

    # # Make the prediction for the last row
    # prediction = knn.predict(new_data)

    # print("Prediction with Library: ", prediction[0])