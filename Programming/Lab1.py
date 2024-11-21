import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Task 1: Develop a Multilayer Neural Network

# Define the neural network architecture
class DeepNeuralNetwork(nn.Module):
    def __init__(self):
        super(DeepNeuralNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output(x)
        return x

# Task 2: Develop a Training Set

# Generate synthetic binary classification data
X, y = make_classification(
    n_samples=1000,        # Number of samples
    n_features=20,         # Number of features
    n_informative=15,      # Informative features
    n_redundant=5,         # Redundant features
    n_classes=2,           # Binary classification
    random_state=42        # Reproducibility
)

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data into PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Instantiate the model
model = DeepNeuralNetwork()

# Define the loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
batch_size = 32
history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

for epoch in range(num_epochs):
    model.train()
    
    # Mini-batch training
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0
    correct_train = 0
    
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        epoch_loss += loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate batch accuracy
        correct_train += ((outputs > 0.5) == batch_y).float().sum().item()

    # Calculate training accuracy
    train_accuracy = correct_train / X_train.size(0)
    history["loss"].append(epoch_loss / (X_train.size(0) / batch_size))
    history["accuracy"].append(train_accuracy)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs, y_test)
        val_accuracy = ((val_outputs > 0.5) == y_test).float().mean().item()
        history["val_loss"].append(val_loss.item())
        history["val_accuracy"].append(val_accuracy)
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history["accuracy"], label="Training Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history["loss"], label="Training Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()
