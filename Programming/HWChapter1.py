# McCulloch-Pitts Neuron Model Simulation
class McCullochPittsNeuron:
    def __init__(self, weights, threshold):
        """
        Initialize the McCulloch-Pitts neuron with weights and a threshold.
        :param weights: List of weights associated with each input
        :param threshold: The threshold value for the neuron to fire (output 1)
        """
        self.weights = weights
        self.threshold = threshold

    def activate(self, inputs):
        """
        Simulate the neuron by taking inputs and applying the activation function.
        :param inputs: List of binary inputs (0 or 1)
        :return: Output of the neuron (0 or 1)
        """
        # Calculate weighted sum of inputs
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs))

        # Apply threshold to determine if the neuron fires
        if weighted_sum > self.threshold:
            return 1  # Neuron fires (output 1)
        else:
            return 0  # Neuron does not fire (output 0)


# Example usage:
if __name__ == "__main__":
    # Define the weights for each input
    weights = [1, 1, -1]  # Example: positive, positive, and inhibitory input

    # Define the threshold
    threshold = 1

    # Create the McCulloch-Pitts neuron
    neuron = McCullochPittsNeuron(weights, threshold)

    # Define a set of inputs (binary)
    inputs = [1, 1, 0]  # Example: two excitatory inputs, one inhibitory

    # Get the output of the neuron
    output = neuron.activate(inputs)

    print(f"Input: {inputs} -> Output: {output}")
