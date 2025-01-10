import torch
from torch.nn.modules import Module
from torch.utils.data import DataLoader

import os
import matplotlib.pyplot as plt
import numpy as np

from visualize import compare_predictions

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

def train_model_with_tensorboard(model, device, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir="runs/facial_keypoints")

    val_images, val_keypoints = next(iter(val_loader))
    
    for epoch in range(num_epochs):
        # Oblicz blad walidacji
        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, keypoints in val_loader:
                images.to(device)
                keypoints.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, keypoints)
                val_losses.append(loss.item())

        # RMSE dla bledu walidacji
        val_loss = (sum(val_losses) / len(val_loader) ** .5)

        # Trenowanie modelu
        model.train()
        train_losses = []
        for images, keypoints in train_loader:
            images.to(device)
            keypoints.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, keypoints)

            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()

        # RMSE dla bledu treningowego
        train_loss = (sum(train_losses) / len(train_loader)) ** .5

        # Zmiana kroku treningowego na podstawie bledu walidacji
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

        # Log to TensorBoard
        writer.add_scalar("Train loss", train_loss, epoch)
        writer.add_scalar("Val loss", val_loss, epoch)
        writer.add_figure('Predykcje vs prawidlowe', compare_predictions(model, device, val_images, val_keypoints, 1), global_step=epoch)
            
    writer.close()

