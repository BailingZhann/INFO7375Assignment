import numpy as np

# Step 1: Define the activation functions in a class
class ActivationFunctions:
    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x))  # for numerical stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
# Step 2: Define the class structure for a deep neural network
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.params = self.initialize_params(layers)
        
    def initialize_params(self, layers):
        np.random.seed(1)
        params = {}
        for l in range(1, len(layers)):
            params['W' + str(l)] = np.random.randn(layers[l], layers[l-1]) * 0.01
            params['b' + str(l)] = np.zeros((layers[l], 1))
        return params

    def forward_propagation(self, X):
        A = X
        caches = []
        L = len(self.params) // 2
        
        for l in range(1, L):
            A_prev = A
            Z = np.dot(self.params['W' + str(l)], A_prev) + self.params['b' + str(l)]
            A = ActivationFunctions.relu(Z)
            caches.append((A_prev, Z))
        
        ZL = np.dot(self.params['W' + str(L)], A) + self.params['b' + str(L)]
        AL = ActivationFunctions.softmax(ZL)  # Output layer uses softmax for classification
        caches.append((A, ZL))
        
        return AL, caches
    
    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(AL)) / m
        return np.squeeze(cost)  # ensures cost is a scalar


# Step 3: Implement backpropagation
class NeuralNetwork:
    # Assuming forward_propagation and other methods already implemented

    def backward_propagation(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        
        # Calculate the gradient of the loss with respect to the final output
        dZL = AL - Y
        grads['dW' + str(L)] = np.dot(dZL, caches[L-1][0].T) / m
        grads['db' + str(L)] = np.sum(dZL, axis=1, keepdims=True) / m
        
        # Backpropagate through each layer
        for l in reversed(range(L-1)):
            A_prev, Z = caches[l]
            dA = np.dot(self.params['W' + str(l+2)].T, dZL)
            dZ = np.multiply(dA, np.int64(A_prev > 0))  # Derivative for ReLU
            grads['dW' + str(l+1)] = np.dot(dZ, caches[l][0].T) / m
            grads['db' + str(l+1)] = np.sum(dZ, axis=1, keepdims=True) / m
            dZL = dZ
            
        return grads

# Step 4: Training the neural network
def update_params(self, grads, learning_rate):
    L = len(self.params) // 2
    for l in range(1, L+1):
        self.params['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        self.params['b' + str(l)] -= learning_rate * grads['db' + str(l)]

def train(self, X, Y, learning_rate=0.01, num_iterations=1000):
    for i in range(num_iterations):
        AL, caches = self.forward_propagation(X)
        cost = self.compute_cost(AL, Y)
        grads = self.backward_propagation(AL, Y, caches)
        self.update_params(grads, learning_rate)

        if i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")

