import torch
import torch.nn as nn
import torch.nn.functional as F

class KeypointCNN(nn.Module):
    def __init__(self):
        super(KeypointCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # Wyjście: (32, 96, 96)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Wyjście: (64, 48, 48)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # Wyjście: (128, 24, 24)

        self.pool = nn.MaxPool2d(2, 2) # Redukcja wymiarów o połowę
        self.fc1 = nn.Linear(128 * 12 * 12, 512) # Pełne połączenie
        self.fc2 = nn.Linear(512, 30) # 15 punktów kluczowych (x, y) dla każdego punktu

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1) # Spłaszcz
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x