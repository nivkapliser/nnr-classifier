import time
import json
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import List, Tuple
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score


def preprocess_data(train_data: pd.DataFrame, validation_data: pd.DataFrame,
                    test_data: pd.DataFrame) -> Tuple[np.ndarray, ...]:
    """
    Description:
        Prepares the training, validation, and test datasets by handling categorical columns and scaling
        numeric features for consistent input across datasets.

    Params:
        train_data (pd.DataFrame): The training dataset.
        validation_data (pd.DataFrame): The validation dataset.
        test_data (pd.DataFrame): The test dataset.

    Returns:
        Tuple[np.ndarray, ...]: Processed training, validation, and test datasets as numpy arrays.
    """
    # Create copies to avoid modifying original data
    train = train_data.copy()
    val = validation_data.copy()
    test = test_data.copy()

    # Handle categorical columns
    categorical_columns = train.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()  # we use separate encoder per column for better encoding
        train[col] = le.fit_transform(train[col])
        # use the same encoder for the validation and test sets
        if val is not None:
            val[col] = le.transform(val[col])
        if test is not None:
            test[col] = le.transform(test[col])

    # Scale features column by column for better handling of different scales
    # StandardScaler is used to scale features to have mean=0 and variance=1
    # using zeros_like to handle data alignment with original data
    scaler = StandardScaler()
    X_train = np.zeros_like(train.values, dtype=float)
    if val is not None and test is not None:
        X_val = np.zeros_like(val.values, dtype=float)
        X_test = np.zeros_like(test.values, dtype=float)

        for i in range(train.shape[1]):
            X_train[:, i] = scaler.fit_transform(train.iloc[:, i].values.reshape(-1, 1)).ravel()
            X_val[:, i] = scaler.transform(val.iloc[:, i].values.reshape(-1, 1)).ravel()
            X_test[:, i] = scaler.transform(test.iloc[:, i].values.reshape(-1, 1)).ravel()
        return X_train, X_val, X_test

    return X_train


def calculate_distances_vectorized(X_train: np.ndarray, X_target: np.ndarray) -> np.ndarray:
    """
    Description:
        Computes the pairwise Euclidean distances between training points and target points in a vectorized,
        efficient way.

    Params:
        X_train (np.ndarray): Training dataset features.
        X_target (np.ndarray): Target dataset features.

    Returns:
        np.ndarray: A matrix with distances between rows (train) and cols (target)


    """
    # Compute distances efficiently
    train_squared = np.sum(X_train ** 2, axis=1)[:, np.newaxis]
    target_squared = np.sum(X_target ** 2, axis=1)
    cross_term = np.dot(X_train, X_target.T)

    return np.sqrt(np.maximum(train_squared + target_squared - 2 * cross_term, 0))


def predict(distances: np.ndarray, y_train: np.ndarray, radius: float,
                  default_class: object, min_neighbors: int = 3) -> np.ndarray:
    """
    Description:
        Generates predictions for target points based on nearest neighbors within a given radius.
        Adjusts the radius adaptively if the minimum number of neighbors is not met.

    Params:
        distances (np.ndarray): Distance matrix between training and target points.
        y_train (np.ndarray): Labels of the training dataset.
        radius (float): Radius within which neighbors are considered.
        default_class (object): Default class used when no neighbors are found.
        min_neighbors (int, optional): Minimum required neighbors for a valid prediction. Default is 3 (found it best).

    Returns:
        np.ndarray: Predicted labels for the target points.
    """
    predictions = []
    for i in range(distances.shape[1]):
        neighbors_idx = distances[:, i] <= radius
        neighbor_count = np.sum(neighbors_idx)

        # If too few neighbors, adaptively increase radius for this point
        if neighbor_count < min_neighbors:
            sorted_distances = np.sort(distances[:, i])
            adaptive_radius = sorted_distances[min_neighbors - 1]
            neighbors_idx = distances[:, i] <= adaptive_radius

        if np.any(neighbors_idx):
            neighbor_labels = y_train[neighbors_idx]
            prediction = Counter(neighbor_labels).most_common(1)[0][0]
        else:
            prediction = default_class
        predictions.append(prediction)

    return np.array(predictions)


def find_optimal_radius(X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray) -> float:
    """
    Description:
        Finds the optimal radius for nearest-neighbor classification by maximizing accuracy on validation data.

    Params:
        X_train (np.ndarray): Preprocessed training features.
        y_train (np.ndarray): Labels of the training dataset.
        X_val (np.ndarray): Preprocessed validation features.
        y_val (np.ndarray): Labels of the validation dataset.

    Returns:
        float: Optimal radius value.
    """

    # Calculate initial distances - train-to-validation distances matrix
    distances = calculate_distances_vectorized(X_train, X_val)

    # Define radius range based on distance distribution
    percentiles = np.linspace(0, 40, 40)  # tried many values this works best - 0 for the minimum and 40 for 40% of the distances
    radii = np.percentile(distances, percentiles) # generating radius based on the percentile range

    best_accuracy = 0
    best_radius = radii[0]
    default_class = Counter(y_train).most_common(1)[0][0]

    # Try different radii
    for radius in radii:
        predictions = predict(distances, y_train, radius, default_class)
        accuracy = accuracy_score(y_val, predictions)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_radius = radius

    return best_radius


def classify_with_NNR(data_trn: str, data_vld: str, df_tst: DataFrame) -> List:
    """
    Description:
        Implements the nearest-neighbor radius (NNR) classification pipeline, including data preprocessing,
        radius optimization, and prediction.

    Parameters:
        data_trn (str): Path to the training dataset CSV file.
        data_vld (str): Path to the validation dataset CSV file.
        df_tst (DataFrame): Test dataset.

    Returns:
        List: Predicted labels for the test dataset.
    """

    print(f'Starting classification with {data_trn}, {data_vld}, predicting on {len(df_tst)} instances')

    # Load and prepare data
    train_data = pd.read_csv(data_trn)
    val_data = pd.read_csv(data_vld)

    X_train_raw = train_data.iloc[:, :-1]
    y_train = train_data["class"].values
    X_val_raw = val_data.iloc[:, :-1]
    y_val = val_data["class"].values

    # Preprocess data
    X_train, X_val, X_test = preprocess_data(X_train_raw, X_val_raw, df_tst)

    # Find optimal parameters
    optimal_radius = find_optimal_radius(X_train, y_train, X_val, y_val)

    # Make predictions
    test_distances = calculate_distances_vectorized(X_train, X_test)
    default_class = Counter(y_train).most_common(1)[0][0]
    predictions = list(predict(test_distances, y_train, optimal_radius, default_class))

    return predictions

if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    df = pd.read_csv(config['data_file_test'])
    predicted = classify_with_NNR(config['data_file_train'],
                                  config['data_file_validation'],
                                  df.drop(['class'], axis=1))

    labels = df['class'].values
    if not predicted:  # empty prediction, should not happen in your implementation
        predicted = list(range(len(labels)))

    assert(len(labels) == len(predicted))  # make sure you predict label for all test instances
    print(f'test set classification accuracy: {accuracy_score(labels, predicted)}')

    print(f'total time: {round(time.time()-start, 0)} sec')

