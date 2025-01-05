import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from loadkaggle import load_kaggle
from dataset import FacialKeypointsDataset
from cnn import KeypointCNN
from train import train_model
from visualize import visualize_predictions

# Przygotowanie danych
train_images, train_keypoints = load_kaggle("dane/training.csv")
test_images, test_keypoints = load_kaggle("dane/test.csv")

# Sprawdź obrazy i punkty kluczowe
print(f"Any NaN in train_images: {np.isnan(train_images).any()}")
print(f"Any NaN in train_keypoints: {np.isnan(train_keypoints).any()}")

print(f"Any Inf in train_images: {np.isinf(train_images).any()}")
print(f"Any Inf in train_keypoints: {np.isinf(train_keypoints).any()}")

# Dataset dla treningu
train_dataset = FacialKeypointsDataset(train_images, train_keypoints)

# Dataset dla walidacji
test_dataset = FacialKeypointsDataset(test_images, test_keypoints)

# DataLoader dla treningu i walidacji
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model
model = KeypointCNN()

# Funkcja straty
criterion = nn.MSELoss()

# Optymalizator
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Przeniesienie modelu na GPU (jeśli dostępne)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Trenowanie modelu
train_model(model, device, train_loader, test_loader, criterion, optimizer, 20)

# Wizualizacja wyników
visualize_predictions(model, device, test_loader)