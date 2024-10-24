print("Script started.")
import numpy as np
print("Numpy imported.")
import tensorflow as tf
print("TensorFlow imported.")

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import Adam
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set a random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Task 2: Generate a synthetic dataset for binary classification
# Create a dataset with 1000 samples, 20 features, 10 informative features, and binary labels
input_dim = 20
X, y = make_classification(
    n_samples=1000, 
    n_features=input_dim, 
    n_informative=10, 
    n_redundant=5, 
    n_clusters_per_class=2, 
    random_state=42
)

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (mean=0, variance=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Task 1: Define a multilayer neural network model
model = Sequential([
    Dense(10, activation='relu', input_shape=(input_dim,)),  # Layer 1: 10 neurons, ReLU
    Dense(8, activation='relu'),                             # Layer 2: 8 neurons, ReLU
    Dense(8, activation='relu'),                             # Layer 3: 8 neurons, ReLU
    Dense(4, activation='relu'),                             # Layer 4: 4 neurons, ReLU
    Dense(1, activation='sigmoid')                           # Layer 5: 1 neuron, Sigmoid
])

# Compile the model with Adam optimizer and binary crossentropy loss
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
print("Model Summary:")
model.summary()

# Train the model and store the training history
print("\nStarting model training...\n")
history = model.fit(
    X_train, 
    y_train, 
    epochs=50, 
    batch_size=32, 
    validation_data=(X_test, y_test), 
    verbose=1
)

# Display training results
print("\nTraining completed.")
print(f"Final training accuracy: {100 * history.history['accuracy'][-1]:.2f}%")
print(f"Final validation accuracy: {100 * history.history['val_accuracy'][-1]:.2f}%")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {100 * test_accuracy:.2f}%")
