import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

# Scale the features for better neural network performance
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define the neural network architecture exactly as per lab specification with slight dropout regularization
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Dropout(0.2)
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

# Instantiate the model, define loss and optimizer
model = NeuralNetwork()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Reduced initial learning rate

# Set up learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

# Train the model with a smaller batch size
num_epochs = 100
batch_size = 16  # Small batch size for potentially better generalization
history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

for epoch in range(num_epochs):
    model.train()
    
    # Mini-batch training
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0
    correct_train = 0
    
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        epoch_loss += loss.item()
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate batch accuracy
        correct_train += ((outputs > 0.5) == batch_y).float().sum().item()

    scheduler.step()
    
    # Calculate epoch-level training accuracy and loss
    train_accuracy = correct_train / X_train.size(0)
    history["accuracy"].append(train_accuracy)
    history["loss"].append(epoch_loss / (X_train.size(0) / batch_size))
    
    # Validation
    with torch.no_grad():
        model.eval()
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_accuracy = ((test_outputs > 0.5) == y_test).float().mean().item()
        history["val_accuracy"].append(test_accuracy)
        history["val_loss"].append(test_loss.item())
        
    # Print loss and accuracy
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Plot training history
plt.figure(figsize=(12, 4))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
