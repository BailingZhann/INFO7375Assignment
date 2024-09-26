import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create random grayscale 20x20 images for digits 0-9
def generate_random_digit_images():
    data = {}
    for digit in range(10):
        images = [np.random.rand(20, 20) for _ in range(10)]
        data[digit] = images
    return data

# Step 2: Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Step 3: Derivative of Sigmoid for backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)

# Step 4: Initialize weights and bias
def initialize_parameters(input_size, output_size):
    weights = np.random.rand(input_size, output_size) - 0.5
    bias = np.random.rand(output_size) - 0.5
    return weights, bias

# Step 5: Forward propagation
def forward_propagation(X, weights, bias):
    Z = np.dot(X, weights) + bias
    A = sigmoid(Z)
    return A

# Step 6: Loss function (Mean Squared Error)
def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Step 7: Backpropagation and gradient descent update
def backpropagation(X, y_true, y_pred, weights, bias, learning_rate):
    m = y_true.shape[0]  # number of samples
    
    # Calculate error
    dA = y_pred - y_true
    dZ = dA * sigmoid_derivative(y_pred)
    
    # Update weights and bias
    weights -= learning_rate * np.dot(X.T, dZ) / m
    bias -= learning_rate * np.sum(dZ, axis=0) / m
    
    return weights, bias

# Step 8: Train the Perceptron
def train_perceptron(X_train, y_train, input_size, output_size, epochs, learning_rate):
    weights, bias = initialize_parameters(input_size, output_size)
    
    for epoch in range(epochs):
        # Forward pass
        y_pred = forward_propagation(X_train, weights, bias)
        
        # Compute loss
        loss = compute_loss(y_train, y_pred)
        print(f"Epoch {epoch+1}, Loss: {loss}")
        
        # Backpropagation and update
        weights, bias = backpropagation(X_train, y_train, y_pred, weights, bias, learning_rate)
    
    return weights, bias

# Step 9: Test the Perceptron
def test_perceptron(X_test, weights, bias):
    y_pred = forward_propagation(X_test, weights, bias)
    return y_pred

# Main Execution:
if __name__ == "__main__":
    # Generate random digit images
    data = generate_random_digit_images()
    
    # Prepare training data: Flatten the 20x20 images and create labels
    X_train = np.array([img.flatten() for digit in data for img in data[digit]])  # Shape: (100, 400)
    y_train = np.array([[1 if i == digit else 0 for i in range(10)] for digit in range(10) for _ in range(10)])  # One-hot labels
    
    # Set parameters
    input_size = 400  # 20x20 pixels
    output_size = 10  # 10 classes (digits 0-9)
    epochs = 1000
    learning_rate = 0.1
    
    # Train the perceptron
    weights, bias = train_perceptron(X_train, y_train, input_size, output_size, epochs, learning_rate)
    
    # Prepare some test data (unlabeled)
    X_test = np.random.rand(5, 400)  # Generate 5 random test images of size 20x20
    
    # Test the perceptron
    y_pred = test_perceptron(X_test, weights, bias)
    print("Predictions: ", y_pred)
    
    # You can visualize the test images if needed
    for i, img in enumerate(X_test):
        plt.imshow(img.reshape(20, 20), cmap="gray")
        plt.title(f"Predicted: {np.argmax(y_pred[i])}")
        plt.show()
