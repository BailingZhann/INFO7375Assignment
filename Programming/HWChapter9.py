import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    def __init__(self, n_x, n_h, n_y, learning_rate=0.01, lambd=0, keep_prob=1.0):
        """
        Initialize the neural network.

        Arguments:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer
        learning_rate -- learning rate for gradient descent
        lambd -- regularization parameter (L2 regularization)
        keep_prob -- probability of keeping a neuron active during dropout
        """
        self.parameters = {}
        self.learning_rate = learning_rate
        self.lambd = lambd  # L2 regularization parameter
        self.keep_prob = keep_prob  # Dropout keep probability
        self.initialize_parameters(n_x, n_h, n_y)
        self.cache = {}
        self.grads = {}
        
    def initialize_parameters(self, n_x, n_h, n_y):
        """
        Initialize parameters with small random values.

        Arguments:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer
        """
        np.random.seed(1)
        self.parameters['W1'] = np.random.randn(n_h, n_x) * 0.01
        self.parameters['b1'] = np.zeros((n_h, 1))
        self.parameters['W2'] = np.random.randn(n_y, n_h) * 0.01
        self.parameters['b2'] = np.zeros((n_y, 1))
    
    def forward_propagation(self, X):
        """
        Implements the forward propagation with optional dropout.

        Arguments:
        X -- input data of size (n_x, m)

        Returns:
        A2 -- The sigmoid output of the second activation
        """
        W1 = self.parameters['W1']
        b1 = self.parameters['b1']
        W2 = self.parameters['W2']
        b2 = self.parameters['b2']
        
        # LINEAR -> TANH -> [Dropout] -> LINEAR -> SIGMOID
        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        
        # Implement dropout
        if self.keep_prob < 1.0:
            D1 = np.random.rand(A1.shape[0], A1.shape[1]) < self.keep_prob
            A1 = np.multiply(A1, D1)
            A1 /= self.keep_prob  # Inverted dropout scaling
            self.cache['D1'] = D1
        else:
            self.cache['D1'] = None
        
        Z2 = np.dot(W2, A1) + b2
        A2 = self.sigmoid(Z2)
        
        # Save values for backward propagation
        self.cache['Z1'] = Z1
        self.cache['A1'] = A1
        self.cache['Z2'] = Z2
        self.cache['A2'] = A2
        
        return A2
    
    def compute_cost(self, A2, Y):
        """
        Computes the cross-entropy cost with L2 regularization.

        Arguments:
        A2 -- The sigmoid output of the second activation
        Y -- "true" labels vector of shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """
        m = Y.shape[1]
        
        # Compute cross-entropy cost
        logprobs = np.multiply(-np.log(A2 + 1e-8), Y) + np.multiply(-np.log(1 - A2 + 1e-8), 1 - Y)
        cross_entropy_cost = (1./m) * np.sum(logprobs)
        
        # Add L2 regularization cost
        if self.lambd != 0:
            W1 = self.parameters['W1']
            W2 = self.parameters['W2']
            L2_regularization_cost = (self.lambd/(2*m)) * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
            cost = cross_entropy_cost + L2_regularization_cost
        else:
            cost = cross_entropy_cost
        
        return cost
    
    def backward_propagation(self, X, Y):
        """
        Implements the backward propagation with L2 regularization and dropout.

        Arguments:
        X -- input data of shape (n_x, m)
        Y -- "true" labels vector of shape (1, number of examples)
        """
        m = X.shape[1]
        W1 = self.parameters['W1']
        W2 = self.parameters['W2']
        A1 = self.cache['A1']
        A2 = self.cache['A2']
        D1 = self.cache['D1']
        
        # Backward propagation
        dZ2 = A2 - Y
        dW2 = (1./m) * np.dot(dZ2, A1.T)
        db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)
        
        # Add L2 regularization to gradients
        if self.lambd != 0:
            dW2 += (self.lambd/m) * W2
        
        dA1 = np.dot(W2.T, dZ2)
        
        # Apply dropout mask to dA1
        if self.keep_prob < 1.0 and D1 is not None:
            dA1 = np.multiply(dA1, D1)
            dA1 /= self.keep_prob  # Inverted dropout scaling
        
        dZ1 = dA1 * (1 - np.power(A1, 2))  # derivative of tanh activation
        
        dW1 = (1./m) * np.dot(dZ1, X.T)
        db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)
        
        # Add L2 regularization to gradients
        if self.lambd != 0:
            dW1 += (self.lambd/m) * W1
        
        # Update gradients
        self.grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    
    def update_parameters(self):
        """
        Updates parameters using the gradient descent update rule.
        """
        self.parameters['W1'] -= self.learning_rate * self.grads['dW1']
        self.parameters['b1'] -= self.learning_rate * self.grads['db1']
        self.parameters['W2'] -= self.learning_rate * self.grads['dW2']
        self.parameters['b2'] -= self.learning_rate * self.grads['db2']
    
    def train(self, X, Y, num_iterations=10000, print_cost=False):
        """
        Trains the neural network.

        Arguments:
        X -- input data of shape (n_x, m)
        Y -- true labels of shape (1, m)
        num_iterations -- number of iterations to train
        print_cost -- if True, print the cost every 1000 iterations
        """
        for i in range(num_iterations):
            A2 = self.forward_propagation(X)
            cost = self.compute_cost(A2, Y)
            self.backward_propagation(X, Y)
            self.update_parameters()
            if print_cost and i % 1000 == 0:
                print(f"Cost after iteration {i}: {cost}")
    
    def predict(self, X):
        """
        Predicts outputs for input data X.

        Arguments:
        X -- input data of shape (n_x, m)

        Returns:
        predictions -- predictions vector
        """
        A2 = self.forward_propagation(X)
        predictions = (A2 > 0.5)
        return predictions.astype(int)
    
    @staticmethod
    def sigmoid(z):
        """
        Implements the sigmoid activation function.

        Arguments:
        z -- numpy array of any shape

        Returns:
        s -- sigmoid(z)
        """
        return 1 / (1 + np.exp(-z))

def main():
    """
    Main function to run the neural network with regularization and dropout.
    """
    # Generate a dataset
    X, Y = make_moons(n_samples=1000, noise=0.2)
    X = X.T
    Y = Y.reshape(1, Y.shape[0])

    # Split the dataset into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X.T, Y.T, test_size=0.2)
    X_train = X_train.T
    X_test = X_test.T
    Y_train = Y_train.T
    Y_test = Y_test.T

    # Define network dimensions
    n_x = X_train.shape[0]  # Number of input features
    n_h = 5  # Number of hidden units
    n_y = Y_train.shape[0]  # Number of output units

    # Initialize the neural network with L2 regularization and dropout
    nn = NeuralNetwork(n_x, n_h, n_y, learning_rate=0.01, lambd=0.7, keep_prob=0.8)

    # Train the neural network
    nn.train(X_train, Y_train, num_iterations=10000, print_cost=True)

    # Predict on the test set
    predictions = nn.predict(X_test)

    # Calculate accuracy
    accuracy = np.mean(predictions == Y_test) * 100
    print(f'Accuracy on test set: {accuracy:.2f}%')

if __name__ == "__main__":
    main()


