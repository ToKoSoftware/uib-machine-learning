import matplotlib.pyplot as plt


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
