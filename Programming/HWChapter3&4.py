from PIL import Image, ImageDraw
import numpy as np
import random
import matplotlib.pyplot as plt
import os

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function (for backpropagation)
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Loss function (binary cross-entropy)
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Perceptron class
class Perceptron:
    def __init__(self, input_size):
        # Initialize weights and bias randomly
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        
    # Forward propagation
    def forward(self, X):
        self.output = sigmoid(np.dot(X, self.weights) + self.bias)
        return self.output
    
    # Backpropagation and gradient descent
    def backward(self, X, y_true, learning_rate=0.01):
        error = self.output - y_true
        d_weights = np.dot(X.T, error) / X.shape[0]
        d_bias = np.mean(error)
        
        # Update weights and bias
        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias

# Generate a synthetic image of a handwritten digit
def create_digit_image(digit, image_size=(20, 20), variation=0):
    img = Image.new('L', image_size, color=255)  # Create a white grayscale image
    draw = ImageDraw.Draw(img)
    
    # Randomly alter the position or size of the digit for variation
    font_size = random.randint(12, 16) + variation
    x_offset = random.randint(0, 5)
    y_offset = random.randint(0, 5)
    
    # Use a default built-in font (could change to actual handwritten-like font if available)
    draw.text((x_offset, y_offset), str(digit), fill=0)
    
    return img

# Create 10 variations of each digit (0-9)
def generate_digit_dataset():
    dataset = []
    labels = []
    
    for digit in range(10):
        for variation in range(10):
            img = create_digit_image(digit, variation=variation)
            dataset.append(np.array(img))  # Convert image to numpy array
            labels.append(digit)
    
    return np.array(dataset), np.array(labels)

# Save images for visualization in the existing "images" folder (overwrite existing files)
def save_images(dataset, labels, folder_name="/Users/bailingzhan/Desktop/INFO7375 Assignments/Programming/images"):
    # Check if the folder exists
    if os.path.exists(folder_name):
        for i, (img, label) in enumerate(zip(dataset, labels)):
            file_name = f"{folder_name}/digit_{label}_{i}.png"
            # Save the image, overwriting if it exists
            img = Image.fromarray(img)
            img.save(file_name)
            print(f"Saved {file_name}")
    else:
        print(f"The folder '{folder_name}' does not exist. Please create the folder before running the code.")

# Generate and save images to the existing 'images' folder
dataset, labels = generate_digit_dataset()
save_images(dataset, labels, folder_name="/Users/bailingzhan/Desktop/INFO7375 Assignments/Programming/images")

# Flatten the 20x20 images for the Perceptron input
X_train = dataset.reshape(100, -1)  # 100 images of size 20x20 flattened to 400 pixels
y_train_binary = np.where(labels == 0, 1, 0)  # Example: Training only for digit 0

# Initialize Perceptron with 400 input features (for 20x20 images)
perceptron = Perceptron(input_size=400)

# Train the Perceptron
def train_perceptron(perceptron, X_train, y_train, epochs=10, learning_rate=0.01):
    for epoch in range(epochs):
        for i in range(X_train.shape[0]):
            output = perceptron.forward(X_train[i])
            perceptron.backward(X_train[i], y_train[i], learning_rate)

# Train the Perceptron
train_perceptron(perceptron, X_train, y_train_binary)

# Test the perceptron
def test_perceptron(perceptron, X_test):
    predictions = []
    for x in X_test:
        prediction = perceptron.forward(x.reshape(-1))
        predictions.append(prediction)
    return predictions

# Generate unlabeled test images
def generate_test_images(num_images=10):
    test_images = []
    
    for _ in range(num_images):
        digit = random.randint(0, 9)
        img = create_digit_image(digit)
        test_images.append(np.array(img))  # Store only images, no labels
    
    return np.array(test_images)

# Save test images for visualization in the existing "images" folder (overwrite existing files)
def save_test_images(test_images, folder_name="/Users/bailingzhan/Desktop/INFO7375 Assignments/Programming/images"):
    # Check if the folder exists
    if os.path.exists(folder_name):
        for i, img in enumerate(test_images):
            file_name = f"{folder_name}/test_image_{i}.png"
            # Save the image, overwriting if it exists
            img = Image.fromarray(img)
            img.save(file_name)
            print(f"Saved {file_name}")
    else:
        print(f"The folder '{folder_name}' does not exist. Please create the folder before running the code.")

# Generate and save test images to the existing 'images' folder
test_images = generate_test_images()
save_test_images(test_images, folder_name="/Users/bailingzhan/Desktop/INFO7375 Assignments/Programming/images")

# Flatten test images for perceptron testing
X_test = test_images.reshape(10, -1)

# Predict using trained perceptron
predictions = test_perceptron(perceptron, X_test)

# Display test results with predictions
for i, (image, prediction) in enumerate(zip(test_images, predictions)):
    plt.imshow(image, cmap='gray')
    plt.title(f'Prediction: {round(prediction)}')
    plt.show()
