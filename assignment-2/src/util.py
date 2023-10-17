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
