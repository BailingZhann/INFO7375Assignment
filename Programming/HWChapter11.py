import numpy as np

def compute_cost(X, y, theta):
    """
    Compute the cost for linear regression using the mean squared error.
    
    Args:
    X (numpy.ndarray): Input features (with bias term).
    y (numpy.ndarray): Target values.
    theta (numpy.ndarray): Model parameters (weights).
    
    Returns:
    float: The computed cost.
    """
    m = len(y)  # Number of training examples
    predictions = X.dot(theta)  # Predictions of hypothesis
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))  # Mean squared error
    return cost

def mini_batch_gradient_descent(X, y, theta, learning_rate=0.01, batch_size=32, epochs=100):
    """
    Perform mini-batch gradient descent optimization.
    
    Args:
    X (numpy.ndarray): Input features (with bias term).
    y (numpy.ndarray): Target values.
    theta (numpy.ndarray): Initial model parameters.
    learning_rate (float): Learning rate for gradient updates.
    batch_size (int): Number of samples per mini-batch.
    epochs (int): Number of passes over the entire dataset.
    
    Returns:
    numpy.ndarray: Optimized model parameters.
    list: History of the cost at each epoch.
    """
    m = len(y)  # Number of training examples
    cost_history = []  # To store cost at each epoch
    
    for epoch in range(epochs):
        # Shuffle the data before creating mini-batches
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Process mini-batches
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]
            
            # Calculate predictions and gradients
            predictions = X_batch.dot(theta)
            errors = predictions - y_batch
            gradient = (1 / batch_size) * X_batch.T.dot(errors)
            
            # Update parameters
            theta -= learning_rate * gradient
        
        # Calculate and store the cost after each epoch
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        
        # Optionally print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Cost = {cost}")
    
    return theta, cost_history

# Example usage
if __name__ == "__main__":
    # Sample data (features and target values)
    X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
    y = np.array([2, 2.5, 3, 3.5, 4])
    
    # Initial model parameters (weights)
    theta = np.random.randn(2)
    
    # Apply mini-batch gradient descent
    optimized_theta, cost_history = mini_batch_gradient_descent(X, y, theta, learning_rate=0.01, batch_size=2, epochs=100)
    
    print("Optimized Parameters:", optimized_theta)
