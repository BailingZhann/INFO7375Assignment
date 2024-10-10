import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

# Load the dataset (In this case, MNIST)
(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

# Normalize pixel values (0 to 255) to (0 to 1)
x_train_full = x_train_full.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Splitting the full training set into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)

# Further split test set into smaller prototype sets if needed (optional step)
x_test, _, y_test, _ = train_test_split(x_test, y_test, test_size=0.9, random_state=42)  # Limiting test data for prototype

# Print the shapes of the resulting sets to verify splits
print("Training set shape:", x_train.shape, y_train.shape)
print("Validation set shape:", x_val.shape, y_val.shape)
print("Testing set shape:", x_test.shape, y_test.shape)






