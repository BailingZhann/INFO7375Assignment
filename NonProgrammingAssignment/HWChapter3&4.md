# HW to Chapter 3 & 4 “The Perceptron for Logistic Regression“
## Non-programming Assignment:
### 1. Describe the logistic regression

Logistic regression is a statistical model used for binary classification tasks, where the output is either 0 or 1. Instead of predicting continuous values like linear regression, logistic regression models the probability that a given input belongs to a particular class. It uses the sigmoid function to map any real-valued input into the range of 0 to 1, interpreting the output as the probability of belonging to a class.

The equation for logistic regression is:

P(y = 1 | X) = σ(w ⋅ X + b)

Where:

- P(y = 1 | X) is the probability that y equals 1 given the input X.
- σ represents the sigmoid function.
- w is the weight vector.
- X is the input vector.
- b is the bias term.
- w ⋅ X represents the dot product of the weight vector and the input vector.

### 2. How are grayscale and color (RGB) images presented as inputs for the perceptron?

- **Grayscale images:** Grayscale images are represented as 2D arrays of pixel intensities, where each pixel has a single intensity value between 0 (black) and 255 (white). These pixel values can be flattened into a 1D vector to be fed into a perceptron.

- **Color (RGB) images:** Color images are typically represented as 3D arrays, with three channels (Red, Green, Blue). Each channel is a 2D array of pixel intensities ranging from 0 to 255. Similar to grayscale images, the 3D array can be flattened into a 1D vector where the perceptron treats each pixel intensity (from all channels) as a separate input.

### 3. Is image recognition a logistic regression problem? Why?

**No,** image recognition is typically not a logistic regression problem. Logistic regression is primarily used for binary classification (e.g., yes/no decisions). While logistic regression can be used in image recognition to classify between two classes, more complex models like neural networks or convolutional neural networks (CNNs) are better suited for image recognition because images often contain non-linear relationships, multiple classes, and large amounts of data.

### 4. Is home prices prediction a logistic regression problem? Why?

**No,** home price prediction is not a logistic regression problem. Home prices are continuous values (regression task) rather than discrete binary outputs. Linear regression or more advanced regression models like decision trees, random forests, or gradient boosting machines are typically used for home price prediction because the goal is to predict a continuous variable, not classify a binary outcome.

### 5. Is image diagnostics a logistic regression problem? Why?

**It depends.** If image diagnostics refers to classifying images (such as medical scans) into two categories (e.g., healthy vs. diseased), then it can be framed as a logistic regression problem. However, for more complex diagnostic tasks involving multiple classes or intricate patterns, deep learning models (like CNNs) are typically used because they can capture the complex relationships present in images.

### 6. How does gradient descent optimization work?

Gradient descent is an optimization algorithm used to minimize the loss function by iteratively adjusting the weights and biases of a model. The main idea is to update the parameters in the direction of the negative gradient of the loss function with respect to the parameters.

The update rule for gradient descent is:

Where:

w = w − η (∂L / ∂w)


- w is the parameter (weight) being updated.
- η (eta) is the learning rate.
- ∂L / ∂w is the partial derivative of the loss function L with respect to the weight w.

Gradient descent continues until the algorithm converges to a minimum, or a stopping criterion is reached.

### 7. How does image recognition work as logistic regression classifier?

For binary image recognition using logistic regression:

1. The image is flattened into a 1D vector of pixel intensities.
2. Logistic regression applies a weighted sum to these pixel values, followed by a sigmoid function, to compute the probability that the image belongs to a certain class.
3. The output is a probability between 0 and 1. If the probability exceeds a certain threshold (usually 0.5), the image is classified into one class, otherwise into another.

However, because image data is often non-linear and high-dimensional, logistic regression is rarely used for complex image recognition tasks. Neural networks or CNNs are better suited for such tasks.

### 8. Describe the logistic regression loss function and explain the reasons behind this choice.

The loss function used in logistic regression is the binary cross-entropy (also called log-loss). The goal of the loss function is to measure the difference between the predicted probabilities and the actual binary labels (0 or 1).

The binary cross-entropy loss is defined as:

Where:

L(y, ŷ) = −(y * log(ŷ) + (1 − y) * log(1 − ŷ))

- y is the actual label (0 or 1).
- ŷ (y-hat) is the predicted probability.
- L is the loss.

The reasons for using binary cross-entropy:

- It penalizes incorrect predictions, especially when the model is confident but wrong.
- It leads to faster and more stable convergence when training using gradient descent.
- Cross-entropy is aligned with the probabilistic interpretation of logistic regression, making it a natural choice for this model.

### 9. Describe the sigmoid activation function and the reasons behind its choice.

The sigmoid activation function maps real-valued inputs to a range between 0 and 1, making it suitable for binary classification problems. The function is defined as:

Where:

σ(z) = 1 / (1 + e^(-z))


- z is the weighted sum of inputs.
- e is the base of the natural logarithm.

Reasons for its choice:
- **Probabilistic interpretation:** The sigmoid output can be interpreted as the probability that a given input belongs to a particular class (since the output is between 0 and 1).
- **Smooth gradient:** The sigmoid function is smooth, making it differentiable everywhere, which is essential for training models using gradient-based optimization techniques like gradient descent.
- **S-shaped curve:** The function compresses large positive or negative values into a small range (near 1 or 0), which helps to model binary outcomes effectively.

However, the sigmoid function has some limitations (e.g., vanishing gradient problem for very large or very small inputs), and for more complex neural networks, other activation functions like ReLU are often preferred.

