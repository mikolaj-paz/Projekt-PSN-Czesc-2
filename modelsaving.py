import torch

def save_model_weights(model, file_path="model_weights.pth"):
    torch.save(model.state_dict(), file_path)
    print(f"Zapisano wagi do pliku {file_path}")

def load_model_weights(model, file_path="model_weights.pth"):
    try:
        model.load_state_dict(torch.load(file_path, weights_only=True))
        print(f"Wczytano wagi z pliku {file_path}")
    except FileNotFoundError:
        print(f"Nie znaleziono pliku {file_path}")