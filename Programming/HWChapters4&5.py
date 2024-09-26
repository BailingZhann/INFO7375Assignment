# Step 1: Define the Activation Class
# The activation function (ReLU, sigmoid, etc.) will be implemented here.
import numpy as np

class Activation:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def sigmoid_derivative(z):
        return Activation.sigmoid(z) * (1 - Activation.sigmoid(z))
    
    @staticmethod
    def relu(z):
        return np.maximum(0, z)
    
    @staticmethod
    def relu_derivative(z):
        return np.where(z > 0, 1, 0)

# Step 2: Define the Neuron Class
# A Neuron will manage weights, bias, and activation functions.
class Neuron:
    def __init__(self, input_size, activation_function, activation_function_derivative):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative  # Add this

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(self.weights, inputs) + self.bias
        self.output = self.activation_function(self.z)
        return self.output

    def backward(self, delta, learning_rate):
        # Backpropagate error and update weights
        dz = delta * self.activation_function_derivative(self.z)  # Use the derivative here
        self.weights -= learning_rate * dz * self.inputs
        self.bias -= learning_rate * dz
        return dz * self.weights

# Step 3: Define the Layer Class
# A Layer will consist of multiple neurons.
class Layer:
    def __init__(self, num_neurons, input_size, activation_function, activation_function_derivative):
        self.neurons = [Neuron(input_size, activation_function, activation_function_derivative) for _ in range(num_neurons)]

    def forward(self, inputs):
        self.inputs = inputs
        return np.array([neuron.forward(inputs) for neuron in self.neurons])

    def backward(self, delta, learning_rate):
        # Backpropagate error through all neurons
        new_delta = np.zeros(self.inputs.shape)
        for i, neuron in enumerate(self.neurons):
            new_delta += neuron.backward(delta[i], learning_rate)
        return new_delta

# Step 4: Define the Parameters Class
# This class will hold parameters like learning rate and others.
class Parameters:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        
        
# Step 5: Model Class
# This class initializes and defines the architecture of the neural network.
class Model:
    def __init__(self, input_size, hidden_layer_size, output_size):
        self.layers = []
        # Hidden layer with sigmoid activation
        self.layers.append(Layer(hidden_layer_size, input_size, Activation.sigmoid, Activation.sigmoid_derivative))
        # Output layer with sigmoid activation
        self.layers.append(Layer(output_size, hidden_layer_size, Activation.sigmoid, Activation.sigmoid_derivative))
    
    def forward(self, inputs):
        activation = inputs
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation

    def backward(self, loss_grad, learning_rate):
        delta = loss_grad
        for layer in reversed(self.layers):
            delta = layer.backward(delta, learning_rate)

# Step 6: Define the LossFunction Class
# The loss function will calculate how far the predictions are from the true values.
class LossFunction:
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mean_squared_error_derivative(y_true, y_pred):
        return y_pred - y_true

# Step 7: Define the ForwardProp Class
# Forward propagation processes the input through the layers.
class ForwardProp:
    def __init__(self, model):
        self.model = model
    
    def forward(self, inputs):
        return self.model.forward(inputs)

# Step 8: Define the BackProp Class
# Backpropagation computes gradients and adjusts weights.
class BackProp:
    def __init__(self, model):
        self.model = model
    
    def backward(self, y_true, y_pred, learning_rate):
        # Compute loss gradient (mean squared error derivative)
        loss_grad = LossFunction.mean_squared_error_derivative(y_true, y_pred)
        # Perform backpropagation
        self.model.backward(loss_grad, learning_rate)
        
# Step 9: GradDescent Class
# Updates the model weights based on gradients (this is already handled inside the Neuron and Layer classes during backpropagation).

# Step 10: Training Class
# Handles the entire training process, combining forward propagation, backpropagation, and gradient descent.
class Training:
    def __init__(self, model, learning_rate, epochs):
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self, X_train, y_train):
        for epoch in range(self.epochs):
            epoch_loss = 0
            for x, y in zip(X_train, y_train):
                # Forward Propagation
                output = self.model.forward(x)
                
                # Calculate loss
                loss = LossFunction.mean_squared_error(y, output)
                epoch_loss += loss
                
                # Backward Propagation
                backprop = BackProp(self.model)
                backprop.backward(y, output, self.learning_rate)
                
            print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(X_train)}')
            
def main():
    # Define the model
    model = Model(input_size=2, hidden_layer_size=10, output_size=1)

    # XOR dataset (inputs and labels)
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[0], [1], [1], [0]])

    # Create training instance
    training = Training(model, learning_rate=0.1, epochs=100)

    # Train the model
    training.train(X_train, y_train)

if __name__ == "__main__":
    main()





