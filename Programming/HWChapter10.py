import numpy as np

def normalize_data(X):
    """
    Normalize the input data X using z-score normalization.
    
    Args:
    X (numpy.ndarray): The input data matrix where each row is a data point and each column is a feature.
    Returns:
    numpy.ndarray: The normalized data matrix where each feature has a mean of 0 and a standard deviation of 1.
    """
    # Calculate the mean and standard deviation for each feature (column-wise)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    # Avoid division by zero by replacing 0 std with 1 (if a feature is constant)
    std[std == 0] = 1
    
    # Normalize the data: (X - mean) / std
    X_normalized = (X - mean) / std
    
    return X_normalized

# Example usage
if __name__ == "__main__":
    # Example dataset (rows are samples, columns are features)
    X = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
    
    # Normalize the dataset
    X_normalized = normalize_data(X)
    
    # Print the normalized data
    print("Original Data:\n", X)
    print("Normalized Data:\n", X_normalized)
