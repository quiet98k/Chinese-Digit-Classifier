import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=15, dropout_rate=0.5):
        super(LeNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # 6 filters, 5x5 kernel
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 16 filters, 5x5 kernel
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling
        
        # Flattened size
        self.flattened_size = 16 * 13 * 13  # (16 channels, 13x13 spatial size after pooling)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 120)  # 120 hidden units
        self.dropout1 = nn.Dropout(p=dropout_rate)  # Dropout after fc1

        self.fc2 = nn.Linear(120, 84)  # 84 hidden units
        self.dropout2 = nn.Dropout(p=dropout_rate)  # Dropout after fc2

        self.fc3 = nn.Linear(84, num_classes)  # Output layer for 15 classes
    
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = F.relu(self.conv1(x))  
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))  
        x = self.pool2(x)
        
        # Flatten the output tensor
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))  
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))  
        x = self.dropout2(x)

        x = self.fc3(x)  # Output layer
        
        return x

# Hyperparameter grid
param_grid = {
    'learning_rate': [0.001, 0.01],
    'batch_size': [64, 256],
    'dropout_rate': [0.3, 0.5],
    'optimizer': ['sgd', 'adam'],
    'num_epochs': [20, 40]
}
