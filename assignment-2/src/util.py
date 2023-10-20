import matplotlib.pyplot as plt
import numpy as np


def visualize_image(image):
    plt.imshow(image.reshape(20, 20), cmap="gray")
    plt.show()


def translate_label_to_class(label):
    classes = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "EMPTY",
    ]
    return classes[label]


def generate_confusion_matrix(conf_matrix, file_name, y_test, title):

    # Get unique labels from y_test
    labels = np.unique(y_test)

    # Create a list of class labels corresponding to the unique labels
    class_labels = [translate_label_to_class(label) for label in labels]

    # Plot the confusion matrix
    plt.figure(figsize=(10, 10))
    tick_marks = np.arange(len(class_labels))
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.tab20)
    plt.colorbar()
    _ = plt.xticks(tick_marks, class_labels, rotation=90)
    _ = plt.yticks(tick_marks, class_labels)
    plt.tight_layout()
    plt.savefig("confusion_matrices/" + file_name, dpi=300)
    plt.show()


def downsampling(X, y):
    # Define the desired maximum instances per class
    max_instances_per_class = 1000

    # Initialize lists to store downsampled data
    downsampled_X = []
    downsampled_y = []
    # Iterate through unique classes
    unique_classes = np.unique(y)
    for class_label in unique_classes:
        # Get indices of instances for this class
        class_indices = np.where(y == class_label)[0]

        # Randomly select up to max_instances_per_class instances for this class
        if len(class_indices) > max_instances_per_class:
            selected_indices = np.random.choice(
                class_indices, max_instances_per_class, replace=False
            )
        else:
            selected_indices = class_indices

        # Append the selected instances to the downsampled data
        downsampled_X.extend(X[selected_indices])
        downsampled_y.extend(y[selected_indices])

    # Convert the downsampled data to NumPy arrays
    downsampled_X = np.array(downsampled_X)
    downsampled_y = np.array(downsampled_y)

    # Shuffle the downsampled data
    shuffle_indices = np.random.permutation(len(downsampled_X))
    downsampled_X = downsampled_X[shuffle_indices]
    downsampled_y = downsampled_y[shuffle_indices]

    # Print the counts for each class in the downsampled dataset
    unique_classes_downsampled, class_counts_downsampled = np.unique(
        downsampled_y, return_counts=True
    )
    return (
        downsampled_X,
        downsampled_y,
        unique_classes_downsampled,
        class_counts_downsampled,
    )
