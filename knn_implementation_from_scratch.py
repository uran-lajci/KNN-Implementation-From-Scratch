import math
import pandas as pd

def euclidean_distance_for_many_dimensions(p_list, q_list, dimension):
    total_distance = 0
    for d in range(dimension):
        total_distance += pow((p_list[d] - q_list[d]),2)
    return math.sqrt(total_distance)


def calculate_neighbor_distances(train_dataset, test_dataset):
    calculated_distances = {}
    for i in range(len(train_dataset)):
        calculated_distances[i] = euclidean_distance_for_many_dimensions(train_dataset.iloc[i], test_dataset, train_dataset.shape[1])
    return calculated_distances


def get_neighbors(my_dict, k):
    sorted_dict = dict(sorted(my_dict.items(), key=lambda item: item[1]))
    subset = {key: sorted_dict[key] for key in list(sorted_dict)[:k]}
    return subset


def predict_regression(selected_elements, k):
    prediction = sum(selected_elements)/k
    return prediction

if __name__ == "__main__":    
    df = pd.read_csv("dataset1.csv")

    train_dataset = df[["feature1","feature2"]].iloc[0:9]
    train_dataset_target = df[["target"]].iloc[0:9]

    test_dataset = df[["feature1","feature2"]].iloc[9]
    test_dataset_target = df[["target"]].iloc[9]

    neighbors = calculate_neighbor_distances(train_dataset, test_dataset)

    k = 3
    k_niereset_neighbors = get_neighbors(neighbors, k)

    selected_elements = []
    for x in k_niereset_neighbors:
        selected_elements.append(train_dataset_target.iloc[x])
    
    knn_prediction = predict_regression(selected_elements, k)
    print("Prediction with our code: ", float(knn_prediction))