import pandas as pd
from sklearn.model_selection import train_test_split

def split_train_data(path, train_path, val_path):
    # Wczytaj dane
    data = pd.read_csv(path)

    # Usuń wiersze z brakującymi wartościami (opcjonalnie)
    data = data.dropna()

    # Podziel dane na treningowe i walidacyjne
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # Zapisz podzielone dane
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)

if __name__ == "__main__":
    split_train_data("dane/training.csv", "dane/train_split.csv", "dane/val_split.csv")