import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import sys

from loadkaggle import load_kaggle
from dataset import FacialKeypointsDataset
from cnn import ImprovedKeypointCNN
from train import train_model_with_tensorboard
from visualize import visualize_predictions, run_webcam_visualization
from menu import main_menu
from modelsaving import save_model_weights, load_model_weights
from augmentation import transform_train_image, transform_val_image, combined_transform

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Model na GPU (jeśli dostępne)
    model = ImprovedKeypointCNN() # Tworzenie modelu

    if torch.cuda.is_available():
        model.cuda()

    print(f"Urzadzenie: {device}")

    choice = main_menu()

    match choice:

        case "1": # Trening od nowa
            # Przygotowanie danych
            print("Ladowanie danych treningowych...")
            train_images, train_keypoints = load_kaggle("dane/train_split.csv")
            val_images, val_keypoints = load_kaggle("dane/val_split.csv")

            # Sprawdź obrazy i punkty kluczowe
            print(f"Czy NaN w train_images: {np.isnan(train_images).any()}")
            print(f"Czy NaN w train_keypoints: {np.isnan(train_keypoints).any()}")
            print(f"Czy Inf w train_images: {np.isinf(train_images).any()}")
            print(f"Czy Inf w train_keypoints: {np.isinf(train_keypoints).any()}")

            train_dataset = FacialKeypointsDataset(train_images, train_keypoints, transform_train_image, combined_transform)
            val_dataset = FacialKeypointsDataset(val_images, val_keypoints, transform_val_image)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # DataLoader dla treningu
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) # DataLoader dla walidacji
            
            criterion = nn.MSELoss() # Funkcja straty

            optimizer = torch.optim.Adam(model.parameters(), lr=.001) # Optymalizator
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=.5, threshold=.03, min_lr=1e-5) # Scheduler

            # Trenowanie modelu
            print("Rozpoczeto trening od zera...")
            train_model_with_tensorboard(model, device, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10000)
            print("Zapisywanie wynikow...")
            save_model_weights(model) # Zapisanie modelu do pliku

        case "2": # Wczytanie wag z pliku
            print("Wczytywanie wag...")
            load_model_weights(model)

            # Testowanie
            print("Wczytywanie danych testowych...")
            test_images, test_keypoints = load_kaggle("dane/training.csv")
            test_dataset = FacialKeypointsDataset(test_images, test_keypoints, transform_train_image, combined_transform) # Dataset dla testow
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # DataLoader dla testow

            # Wizualizacja wynikow
            visualize_predictions(model, device, test_loader)
        
        case "3": # Wizualizacja na kamerze
            print("Uruchamianie wizualizacji...")
            load_model_weights(model)
            run_webcam_visualization(model, device, scale_factor=1.0)

        case "4": # Wyjscie
            sys.exit()
        
        case _:
            print("Niezrozumialy wybor, sprobuj jeszcze raz")
            main()

# Uruchomienie programu
if __name__ == "__main__":
    main()