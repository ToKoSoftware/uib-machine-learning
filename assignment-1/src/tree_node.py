class TreeNode:
    """
    Class for a node in the decision tree.
    """

    def __init__(self):
        self.feature_index = None  # Index of the feature to split on
        self.threshold = None  # Threshold value for the feature
        self.label = None  # Class label for leaf nodes
        self.left = None  # Left subtree (<= threshold)
        self.right = None  # Right subtree (> threshold)
