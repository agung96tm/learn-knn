import pandas as pd
import numpy as np
from collections import Counter
from math import sqrt


def load_data(file_path, cols, label=None):
    data = pd.read_csv(file_path)
    x = data[cols].values
    y = data[label].values if label else None
    return x, y, data


def euclidean_distance(point1, point2):
    return sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))


def knn_predict(x_train, y_train, test_point, k):
    distances = [(euclidean_distance(test_point, x_train), y_train[i]) for i, x_train in enumerate(x_train)]
    k_nearest = sorted(distances, key=lambda x: x[0])[:k]
    labels = [label for _, label in k_nearest]
    most_common = Counter(labels).most_common(1)
    return most_common[0][0]


def run_knn(source, test_f, out, cols, label, k=3):
    x_train, y_train, source_data = load_data(source, cols, label)
    x_test, _, test_data = load_data(test_f, cols)

    predictions = [knn_predict(x_train, y_train, test_point, k) for test_point in x_test]

    test_data['Predicted_Result'] = predictions

    source_data['Original_Result'] = source_data[label]
    source_data[''] = np.nan
    source_data[''] = np.nan

    combined_data = pd.concat([
        source_data[cols + [label] + [''] + ['']],
        test_data[cols + ['Predicted_Result']]
    ], axis=1)

    combined_data.to_csv(out, index=False)
    print(f"Success Generated and Saved to {out}")


source_file = './results/source.csv'
test_file = './results/test_data.csv'
output_file = './results/result.csv'
feature_columns = ['X1', 'X2']
label_column = 'Result'
k = 3

run_knn(source_file, test_file, output_file, feature_columns, label_column, k)
