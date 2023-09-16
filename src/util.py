import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(data, labels, test_size=0.2, random_state=None):
    """
    Split a dataset into training and validation sets.

    Args:
    data (array-like): The feature data.
    labels (array-like): The corresponding labels or target values.
    test_size (float): The proportion of the dataset to include in the validation set (default is 0.2).
    random_state (int): Seed for random number generation (optional).

    Returns:
    tuple: A tuple containing (X_train, X_val, y_train, y_val).
    """
    X_train, X_val, y_train, y_val = train_test_split(
        data, labels, test_size=test_size, random_state=random_state
    )
    return X_train, X_val, y_train, y_val
