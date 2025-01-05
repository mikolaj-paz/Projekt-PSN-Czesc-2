import torch
from torch.nn.modules import Module
from torch.utils.data import DataLoader

def train_model(model : Module, device : torch.device, train_loader : DataLoader, test_loader : DataLoader, criterion, optimizer, epochs=10):
    model.train() # Ustawienie modelu w tryb treningu

    for epoch in range(epochs):
        running_loss = 0.0

        for images, keypoints in train_loader:
            images, keypoints = images.to(device), keypoints.to(device) # Przenieś na GPU

            # Zerowanie gradientów
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Obliczanie straty
            loss = criterion(outputs, keypoints)

            # Backward pass i aktualizacje wag
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        # Średnia strata na epokę
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")