import torch
import torch.nn as nn

from visualize import compare_predictions

def train_model_with_tensorboard(model, device, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

    val_images, val_keypoints = next(iter(val_loader))
    train_images, train_keypoints = next(iter(train_loader))

    writer.add_graph(model, train_images)
    
    for epoch in range(num_epochs):
        # Oblicz blad walidacji
        model.eval()
        val_losses = []
        val_errors = []
        with torch.no_grad():
            for images, keypoints in val_loader:         
                outputs = model(images)
                
                val_losses.append(criterion(outputs, keypoints).item())
                val_errors.append(torch.mean(torch.abs((keypoints - outputs) / keypoints)).item())

        # RMSE dla bledu walidacji
        val_loss = (sum(val_losses) / len(val_loader)) ** .5

        # Wartosc bledu
        val_error = sum(val_errors) / len(val_errors)

        # Trenowanie modelu
        model.train()
        train_losses = []
        for images, keypoints in train_loader:
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
        writer.add_scalar("Train loss", train_loss, epoch+1)
        writer.add_scalar("Val loss", val_loss, epoch+1)
        writer.add_scalar("Accuracy", 1.0-val_error, epoch+1)
        writer.add_figure('Predykcje vs prawidlowe', compare_predictions(model, device, val_images, val_keypoints, 1), global_step=epoch+1)
        conv_layer_idx = 0 # Licznik warstw konwolucyjnych
        lin_layer_idx = 0 # Licznik warstw zwyklych
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layer_idx += 1
                writer.add_histogram(f"Conv2d[{name}]", module.weight, epoch+1)
            elif isinstance(module, nn.Linear):
                lin_layer_idx += 1
                writer.add_histogram(f"Linear[{name}]", module.weight, epoch+1)
            
    writer.close()
