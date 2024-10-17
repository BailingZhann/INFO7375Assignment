import numpy as np

class Softmax:
    def __init__(self):
        pass

    def forward(self, inputs):
        """
        Perform the forward pass of the softmax activation function.
        
        Args:
        inputs (numpy.ndarray): Input data (logits), a 2D array where each row represents a set of logits.
        
        Returns:
        numpy.ndarray: Softmax probabilities for each class.
        """
        # Shift inputs for numerical stability (prevent large exponentials)
        shifted_inputs = inputs - np.max(inputs, axis=1, keepdims=True)
        
        # Calculate the exponentials
        exp_values = np.exp(shifted_inputs)
        
        # Normalize by dividing with the sum of exponentials for each row
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        return probabilities

# Example usage
if __name__ == "__main__":
    # Example logits for three classes
    logits = np.array([[2.0, 1.0, 0.1],[1.0, 3.0, 0.2]])

    # Create a Softmax object
    softmax = Softmax()

    # Apply softmax activation (forward pass)
    probabilities = softmax.forward(logits)

    # Output the probabilities
    print("Softmax Probabilities:\n", probabilities)
