import numpy as np
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import time

class TreeNode:
    def __init__(self):
        self.feature_index = None  # Index of the feature to split on
        self.threshold = None  # Threshold value for the feature
        self.label = None  # Class label for leaf nodes
        self.left = None  # Left subtree (<= threshold)
        self.right = None  # Right subtree (> threshold)

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
    if impurity_measure == 'gini':
        return calculate_gini_index(y)
    elif impurity_measure == 'entropy':
        return calculate_entropy(y)
    else:
        raise ValueError("Invalid impurity measure submitted! Only the measures 'gini' and 'entropy' are supported!")

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
    # Calculating information gain by substracting impurity measures of both weighted subsets
    information_gain = total_impurity - (left_weight * left_impurity + right_weight * right_impurity)
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
            information_gain = calculate_information_gain(X, y, feature_index, threshold, impurity_measure)
            # When information gain is better than existing "best" information gain -> Update
            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_feature_index = feature_index
                best_threshold = threshold

    return best_feature_index, best_threshold

""""
    Recursively implemented algorithm for "learning" / creating a decision tree

    Parameters:
    X (2D Array): Feature matrix
    y (Array): Class labels
    impurity_measure (String): Type of impurity measure to use ('gini' / 'entropy')
    prune (bool, optional): True or false regarding the question to prune the tree. By default False is set.
    prune_data (tuple, optional): Validation data for pruning. By default None is set.

    Return:
    Root node of the decision tree (Type: TreeNode)
"""
def learn(X, y, impurity_measure, prune=False, prune_data=None):
    # If all labels in the current node are the same (pure node)
    if len(np.unique(y)) == 1:
        # Returning leaf node with the class label 
        leaf = TreeNode()
        leaf.label = y[0]
        return leaf
    # if all features have the same value
    if np.all(X[:, 0] == X[:, 0][0]):
         # Returning leaf node with the class label being the most frequent class in the current node.
        leaf = TreeNode()
        leaf.label = Counter(y).most_common(1)[0][0]
        return leaf
    # If not stopped, Calculating best split for decision tree based on information gain
    best_feature_index, best_threshold = find_best_split(X, y, impurity_measure)

    if best_feature_index is None:
        leaf = TreeNode()
        leaf.label = Counter(y).most_common(1)[0][0]
        return leaf
    # Creating tree node on the basis of calculated best feature and threshold
    node = TreeNode()
    node.feature_index = best_feature_index
    node.threshold = best_threshold
    # Splitting data in left and right subtree
    left_mask = X[:, best_feature_index] <= best_threshold
    right_mask = ~left_mask
    # Recursive call of functions for left and right subtrees for creating child nodes
    node.left = learn(X[left_mask], y[left_mask], impurity_measure, prune, prune_data)
    node.right = learn(X[right_mask], y[right_mask], impurity_measure, prune, prune_data)
    # When pruning is enabled
    if prune:
        # Comparing accuracy of node before pruning
        majority_class = Counter(y).most_common(1)[0][0]
        accuracy_before_prune = np.mean(predict(X, node) == y)

        node_label = node.label if node.label is not None else majority_class
         # Comparing accuracy of node after pruning
        node_accuracy = np.mean(predict(prune_data[0], node) == prune_data[1])
        #  If pruning improves accuracy or maintains the same accuracy, the node is pruned 
        if node_accuracy >= accuracy_before_prune:
            #  Making it like a leaf note
            node.left = None
            node.right = None
            node.label = node_label
    # returns the constructed tree with the root node (-> which recursively defines the decision tree)
    return node

"""
    Predicting the class label for a single sample using the decision tree.
    The decision tree is recursively traversated, until leaf node is reached.
    Parameters:
    node (TreeNode): Current node in the decision tree.
    x (Array): Feature values for a single sample.

    Return:
    Predicted class label (Type: float)
"""
def predict_single(node, x):
    # When leaf node is reached
    if node.label is not None:
        # Return class label
        return node.label
    # Recursive Traversation of tree based on feature value in comparison with the threshold of the node 
    if x[node.feature_index] <= node.threshold:
        return predict_single(node.left, x)
    else:
        return predict_single(node.right, x)

"""
    Predicting  class labels for multiple samples using the decision tree.

    Parameters:
    X (2D Array): Feature matrix for multiple samples.
    tree (TreeNode): Root node of the decision tree.

    Return:
    Predicted class labels for the input samples (Type: Array)
"""
def predict(X, tree):
     # Recursive Traversation of tree till leaf is reached and return of array of predicted class labels
    return np.array([predict_single(tree, x) for x in X])


# Load the dataset
wine_data = pd.read_csv("C:/Users/marwi/OneDrive - Universität Potsdam/Intro_to_ML/Assignment 01/wine_dataset.csv")
# X = wine_data[['citric acid', 'residual sugar', 'pH', 'sulphates', 'alcohol']].values
X = wine_data.iloc[:, :-1].values
# y = wine_data['type'].values
y = wine_data.iloc[:, -1].values



# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train and evaluate our custom decision tree (entropy with pruning)
start_time = time.time()
#tree_custom = learn(X_train, y_train, impurity_measure='gini', prune=True, prune_data=(X_val, y_val))
#tree_custom = learn(X_train, y_train, impurity_measure='gini', prune=False, prune_data=None)
tree_custom = learn(X_train, y_train, impurity_measure='entropy', prune=True, prune_data=(X_val, y_val))
#tree_custom = learn(X_train, y_train, impurity_measure='entropy', prune=False, prune_data=None)
y_val_pred_custom = predict(X_val, tree_custom)
custom_accuracy = accuracy_score(y_val, y_val_pred_custom)
custom_time = time.time() - start_time

# Train and evaluate scikit-learn's DecisionTreeClassifier
start_time = time.time()
tree_sklearn = DecisionTreeClassifier(criterion='gini', random_state=42)
tree_sklearn.fit(X_train, y_train)
y_val_pred_sklearn = tree_sklearn.predict(X_val)
# Accuracy calculation realized by percentage-wise match of input arrays
sklearn_accuracy = accuracy_score(y_val, y_val_pred_sklearn)
sklearn_time = time.time() - start_time

# Print the results
print("Custom Decision Tree:")
print("Accuracy:", custom_accuracy)
print("Time:", custom_time)

print("\nScikit-Learn Decision Tree:")
print("Accuracy:", sklearn_accuracy)
print("Time:", sklearn_time)

import json

def tree_to_json(node, feature_names):
    if node is None:
        return None

    node_dict = {
        "feature_index": node.feature_index,
        "threshold": node.threshold,
        "label": node.label
    }

    if feature_names is not None and node.feature_index is not None:
        feature_name = feature_names[node.feature_index]
        node_dict["feature_name"] = feature_name

    node_dict["left"] = tree_to_json(node.left, feature_names)
    node_dict["right"] = tree_to_json(node.right, feature_names)

    return node_dict


# Retrieve feature names from the dataset
# feature_names = wine_data[['citric acid', 'residual sugar', 'pH', 'sulphates', 'alcohol']].columns.tolist()
feature_names = wine_data.columns[:-1].tolist()

# Visualize the custom decision tree as JSON
custom_tree_json = tree_to_json(tree_custom, feature_names)
# print("Custom Decision Tree (JSON format):")
#print(custom_tree_json)