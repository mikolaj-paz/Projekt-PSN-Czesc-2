import pandas as pd
import numpy as np

def load_kaggle(path : str):
    # Wczytaj dane treningowe
    data = pd.read_csv(path)

    # Usuń wiersze z brakującymi wartościami (opcjonalnie)
    data = data.dropna()

    # Punkty kluczowe
    keypoints = data.drop('Image', axis=1).values

    # Zamień kolumnę 'Image' na macierz liczb
    data['Image'] = data['Image'].apply(lambda img: np.fromstring(img, sep=' ', dtype=np.float32))

    images = np.stack(data['Image'].values)

    # Normalizacja do zakresu [0,1]
    images = images / 255.0

    # Dodanie kanału dla PyTorch
    images = np.reshape(images, (-1, 1, 96, 96))

    return images, keypoints