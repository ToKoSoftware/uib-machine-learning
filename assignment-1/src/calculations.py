from collections import Counter
import numpy as np

"""
    Calculate entropy for a set of samples based on class labels.

    Parameters:
    y (Array) - Class labels

    Return:
    entropy value (type: float)
"""


def calculate_entropy(y):
    total_samples = len(y)
    class_counts = Counter(y)
    entropy = 0.0

    for count in class_counts.values():
        probability = count / total_samples
        entropy -= probability * np.log2(probability)

    return entropy


"""
    Calculate gini index for a set of samples based on class labels.

    Parameters:
    y (Array) - Class labels

    Return: 
    gini value (type: float)
"""


def calculate_gini_index(y):
    total_samples = len(y)
    class_counts = Counter(y)
    gini_index = 1.0

    for count in class_counts.values():
        probability = count / total_samples
        gini_index -= probability**2

    return gini_index


""""
    Calculate impurity for a set of samples based on the specified impurity measure.

    Parameters:
    y (Array): Class labels
    impurity_measure (String): Type of impurity measure ('gini' / 'entropy')

    Return:
    value from executed impurity measure - (type: float)
"""


def calculate_impurity(y, impurity_measure):
    if impurity_measure == "gini":
        return calculate_gini_index(y)
    elif impurity_measure == "entropy":
        return calculate_entropy(y)
    else:
        raise ValueError(
            "Invalid impurity measure submitted! Only the measures 'gini' and 'entropy' are supported!"
        )


"""
    Helper function for find_best_split(...) to calculate the information gain for a potential feature split.
    The information gain quantifies how much information is gained by splitting the data based on a given feature and 
    threshold. A higher information gain shows that the feature is a better choice for splitting the data, which is 
    therefore used in the find_best_split(...) function.

    Parameters:
    X (2D Array): Feature matrix
    y (Two-dimensional Array): Class labels or features in X
    feature_index (int): Index of the feature to split on
    threshold (float): Threshold value for the feature split
    impurity_measure (String): Type of impurity measure to use ('gini' / 'entropy')

    Return:
    Information gain (type: float)
"""


def calculate_information_gain(X, y, feature_index, threshold, impurity_measure):
    # Calculating total impurity of the node before a split is performed
    total_impurity = calculate_impurity(y, impurity_measure)
    # Dividing the data into left and right subsets based on feature_index and threshold:
    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask
    # Calculating chosen impurity for the two subsets
    left_impurity = calculate_impurity(y[left_mask], impurity_measure)
    right_impurity = calculate_impurity(y[right_mask], impurity_measure)
    # Calculating weigth of each subset for calculating information gain
    total_samples = len(y)
    left_weight = np.sum(left_mask) / total_samples
    right_weight = np.sum(right_mask) / total_samples
    # Calculating information gain by subtracting impurity measures of both weighted subsets
    information_gain = total_impurity - (
        left_weight * left_impurity + right_weight * right_impurity
    )
    return information_gain


"""
    Find the best feature split based on information gain when building the Decision tree

    Parameters:
    X (2D Array"): Feature matrix
    y (Array): Class labels for features in X
    impurity_measure (String): Type of impurity measure to use ('gini' / 'entropy')

    Returns:
    Best feature index and threshold (Type: Tuple)
"""


def find_best_split(X, y, impurity_measure):
    # Number of features
    n = X.shape[1]
    best_information_gain = 0
    best_feature_index = None
    best_threshold = None
    # Iterating over all features
    for feature_index in range(n):
        # Calculating unique values of a specific feature in the feature matrix X
        unique_values = np.unique(X[:, feature_index])
        # Computing potential thresholds for splitting the feature
        # -> (middle points between two following unique values: 1 & 2 -> 1,5 as threshold)
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2
        # For each threshold in a set feature
        for threshold in thresholds:
            # Calculating information gain for current feature and threshold
            information_gain = calculate_information_gain(
                X, y, feature_index, threshold, impurity_measure
            )
            # When information gain is better than existing "best" information gain -> Update
            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_feature_index = feature_index
                best_threshold = threshold

    return best_feature_index, best_threshold
