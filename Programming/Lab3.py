import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid

# Set seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Generate synthetic binary classification data
X, y = make_classification(
    n_samples=1000,        
    n_features=20,         
    n_informative=15,      
    n_redundant=5,         
    n_classes=2,           
    random_state=42        
)

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, dropout_rate):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU()
        )
        self.output = nn.Linear(4, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.sigmoid(self.output(x))
        return x

# Hyperparameter grid
param_grid = {
    "learning_rate": [0.01, 0.001, 0.0005],
    "batch_size": [16, 32, 64],
    "dropout_rate": [0.2, 0.3],
    "num_epochs": [50, 100]
}

best_model = None
best_accuracy = 0
best_params = {}

# Grid search
for params in ParameterGrid(param_grid):
    learning_rate = params["learning_rate"]
    batch_size = params["batch_size"]
    dropout_rate = params["dropout_rate"]
    num_epochs = params["num_epochs"]
    
    # Initialize the model, criterion, and optimizer
    model = NeuralNetwork(dropout_rate=dropout_rate)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        accuracy = ((outputs > 0.5) == y_test).float().mean().item()
        
    # Track the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_params = params

# Print the best parameters and accuracy
print("Best Hyperparameters:", best_params)
print(f"Best Validation Accuracy: {best_accuracy:.4f}")

# Plot final training performance
plt.title("Validation Accuracy Across Configurations")
plt.bar(list(range(len(param_grid))), [best_accuracy], color="blue")
plt.xlabel("Configurations")
plt.ylabel("Accuracy")
plt.show()
