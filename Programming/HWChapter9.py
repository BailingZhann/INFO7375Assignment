import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, lambda_l2=0.01):
        # Initialize weights and hyperparameters
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.learning_rate = learning_rate
        self.lambda_l2 = lambda_l2  # L2 regularization strength

    def forward(self, X):
        # Forward pass: Input -> Hidden layer -> Output
        self.hidden = np.dot(X, self.weights_input_hidden)
        self.output = np.dot(self.hidden, self.weights_hidden_output)
        return self.output

    def compute_loss(self, predictions, y):
        # Mean Squared Error Loss
        loss = np.mean(np.square(predictions - y))

        # Add L2 regularization penalty
        l2_penalty = (self.lambda_l2 / 2) * (np.sum(np.square(self.weights_input_hidden)) + np.sum(np.square(self.weights_hidden_output)))
        total_loss = loss + l2_penalty
        return total_loss

    def backward(self, X, y):
        # Compute gradients and update weights
        output_error = self.output - y
        hidden_error = np.dot(output_error, self.weights_hidden_output.T)

        # Gradients for the weights
        grad_weights_hidden_output = np.dot(self.hidden.T, output_error) + self.lambda_l2 * self.weights_hidden_output
        grad_weights_input_hidden = np.dot(X.T, hidden_error) + self.lambda_l2 * self.weights_input_hidden

        # Update weights using gradient descent
        self.weights_hidden_output -= self.learning_rate * grad_weights_hidden_output
        self.weights_input_hidden -= self.learning_rate * grad_weights_input_hidden

# Example usage
if __name__ == "__main__":
    # Define example neural network
    nn = NeuralNetwork(input_size=3, hidden_size=5, output_size=1, learning_rate=0.01, lambda_l2=0.01)

    # Example input (X) and output (y)
    X = np.random.randn(10, 3)
    y = np.random.randn(10, 1)

    # Perform forward and backward pass
    predictions = nn.forward(X)
    loss = nn.compute_loss(predictions, y)
    nn.backward(X, y)

    print("Predictions:\n", predictions)
    print("Loss with L2 Regularization:", loss)

class NeuralNetworkWithDropout:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, dropout_rate=0.5):
        # Initialize weights and hyperparameters
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate  # Dropout rate

    def forward(self, X, training=True):
        # Forward pass with optional dropout
        self.hidden = np.dot(X, self.weights_input_hidden)

        if training:
            # Apply dropout during training
            self.dropout_mask = np.random.rand(*self.hidden.shape) > self.dropout_rate
            self.hidden *= self.dropout_mask  # Zero out some neurons
        else:
            # Scale the outputs during testing
            self.hidden *= (1 - self.dropout_rate)
        
        self.output = np.dot(self.hidden, self.weights_hidden_output)
        return self.output

    def compute_loss(self, predictions, y):
        # Mean Squared Error Loss
        loss = np.mean(np.square(predictions - y))
        return loss

    def backward(self, X, y):
        # Compute gradients and update weights
        output_error = self.output - y
        hidden_error = np.dot(output_error, self.weights_hidden_output.T)

        # Apply dropout mask during backpropagation
        hidden_error *= self.dropout_mask

        # Gradients for the weights
        grad_weights_hidden_output = np.dot(self.hidden.T, output_error)
        grad_weights_input_hidden = np.dot(X.T, hidden_error)

        # Update weights using gradient descent
        self.weights_hidden_output -= self.learning_rate * grad_weights_hidden_output
        self.weights_input_hidden -= self.learning_rate * grad_weights_input_hidden

# Example usage
if __name__ == "__main__":
    # Define example neural network with dropout
    nn_dropout = NeuralNetworkWithDropout(input_size=3, hidden_size=5, output_size=1, learning_rate=0.01, dropout_rate=0.5)

    # Example input (X) and output (y)
    X = np.random.randn(10, 3)
    y = np.random.randn(10, 1)

    # Perform forward and backward pass
    predictions = nn_dropout.forward(X, training=True)
    loss = nn_dropout.compute_loss(predictions, y)
    nn_dropout.backward(X, y)

    print("Predictions with Dropout:\n", predictions)
    print("Loss:", loss)
