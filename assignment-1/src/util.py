def tree_to_json(node, feature_names):
    if node is None:
        return None

    node_dict = {
        "feature_index": node.feature_index,
        "threshold": node.threshold,
        "label": node.label,
    }

    if feature_names is not None and node.feature_index is not None:
        feature_name = feature_names[node.feature_index]
        node_dict["feature_name"] = feature_name

    node_dict["left"] = tree_to_json(node.left, feature_names)
    node_dict["right"] = tree_to_json(node.right, feature_names)

    return node_dict
