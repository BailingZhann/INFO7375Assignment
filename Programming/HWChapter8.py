import numpy as np
import os

def generate_random_images(num_images, image_shape=(28, 28)):
    """
    Generate random data to simulate a set of images.
    
    :param num_images: Number of images to generate.
    :param image_shape: Shape of each image (height, width).
    :return: Numpy array of random images.
    """
    return np.random.rand(num_images, *image_shape)

def split_dataset(data, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split the dataset into training, validation, and test sets.

    :param data: The entire dataset (simulated images).
    :param labels: Corresponding labels for the dataset.
    :param train_ratio: Proportion of data for training set.
    :param val_ratio: Proportion of data for validation set.
    :param test_ratio: Proportion of data for test set.
    :return: Split datasets (train, validation, test) and their labels.
    """

    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    # Shuffle the data
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    # Calculate the number of samples for each set
    num_train = int(train_ratio * len(data))
    num_val = int(val_ratio * len(data))

    # Split the data
    train_data, train_labels = data[:num_train], labels[:num_train]
    val_data, val_labels = data[num_train:num_train + num_val], labels[num_train:num_train + num_val]
    test_data, test_labels = data[num_train + num_val:], labels[num_train + num_val:]

    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)

# Example usage
if __name__ == '__main__':
    num_classes = 3  # For example, 3 classes
    num_images_per_class = 100  # Number of images per class
    image_shape = (28, 28)  # Shape of each "image" (e.g., 28x28 pixels)

    # Simulate a dataset of random images and labels
    data = []
    labels = []
    for class_label in range(num_classes):
        class_images = generate_random_images(num_images_per_class, image_shape)
        data.append(class_images)
        labels.append(np.full(num_images_per_class, class_label))

    # Convert list of arrays into a single NumPy array
    data = np.vstack(data)
    labels = np.hstack(labels)

    # Split the dataset into training, validation, and test sets
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = split_dataset(
        data, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    # Output some details about the splits
    print(f"Training set: {train_data.shape[0]} images")
    print(f"Validation set: {val_data.shape[0]} images")
    print(f"Test set: {test_data.shape[0]} images")
