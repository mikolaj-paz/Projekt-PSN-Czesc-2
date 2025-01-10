import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(model, device, test_loader, num_images=10):
    model.eval()
    with torch.no_grad():
        images, keypoints = next(iter(test_loader))

        images = images.to(device)
        predictions = model(images).cpu().numpy()

        for i in range(num_images):
            img = images[i].cpu().numpy().squeeze()
            true_keypoints = keypoints[i].cpu().numpy().reshape(-1, 2)
            pred = predictions[i].reshape(-1, 2) # (x, y)

            plt.imshow(img, cmap='gray')
            plt.scatter(true_keypoints[:, 0], true_keypoints[:, 1], c='g', s=10)
            plt.scatter(pred[:, 0], pred[:, 1], c='r', s=10)
            plt.show()

def run_webcam_visualization(model, device, face_cascade_path="haarcascade_frontalface_default.xml", scale_factor=1.5):
    # Załaduj klasyfikator Haar
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascade_path)
    if face_cascade.empty():
        print("Blad: Nie znaleziono pliku XML Haar Cascade.")
        return

    model.eval()
    cap = cv2.VideoCapture(1) # Domyślna kamera
    if not cap.isOpened():
        print("Blad: Nie mozna uzyskac dostepu do kamery")
        return
    
    print("Nacisnij 'q' aby wyjsc")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Blad: Nie udalo sie pobrac klatki")
            break
        
        # Przeskalowanie obrazu
        frame_resized = cv2.resize(frame, None, fx=1/scale_factor, fy=1/scale_factor, interpolation=cv2.INTER_AREA)

        # Konwersja na skale szarosci
        gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Wykrywanie twarzy
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Wyciecie twarzy
            face = gray_frame[y:y+h, x:x+w]

            # Normalizacja i zmiana rozmiaru
            face_normalized = face / 255.0 # Skalowanie do zakresu [0, 1]
            face_resized = cv2.resize(face_normalized, (96, 96)) # Zmiena rozmiaru na 96x96

            # Przekształcenie na format PyTorch
            input_tensor = torch.tensor(face_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            # Przewidywanie punktow kluczowych
            with torch.no_grad():
                predictions = model(input_tensor).cpu().numpy().reshape(-1, 2)

            # Skalowanie do rozmiarow wycietej twarzy
            predictions[:, 0] *= w / 96
            predictions[:, 1] *= h / 96
            
            # Debug: Wyświetlanie przewidywań w konsoli
            # print(f"Punkty (normalizowane): {predictions}")

            # Przesuniecie do pozycji twarzy
            predictions[:, 0] += x
            predictions[:, 1] += y

            # Debug: Wyświetlanie przeskalowanych punktów
            # print(f"Punkty (skalowane): {predictions}")

            # Rysowanie punktow kluczowych
            for (px, py) in predictions:
                cv2.circle(frame_resized, (int(px), int(py)), 1, (0, 0, 255), -1)

            # Rysowanie prostokata wokol twarzy
            cv2.rectangle(frame_resized, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 1)
        
        # Wyświetlenie klatki
        cv2.imshow("Detekcja punktow charakterystycznych twarzy", frame_resized)

        # Wyjscie po nacisnieciu 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def compare_predictions(model, device, images, keypoints, img_num=None):
    if isinstance(images, list):
        images = torch.stack(images)
    
    images = images.to(device)
    keypoints = keypoints.to(device)

    idx = img_num if img_num is not None else np.random.randint(len(images))

    img = images[idx].cpu().numpy().squeeze()
    true_keypoints = keypoints[idx].cpu().numpy().reshape(-1, 2)

    model.eval()
    with torch.no_grad():
        predictions = model(images).cpu().numpy()
    
    pred = predictions[idx].reshape(-1, 2)

    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.scatter(true_keypoints[:, 0], true_keypoints[:, 1], c='g', s=10)
    ax.scatter(pred[:, 0], pred[:, 1], c='r', s=10)
    ax.axis('off')

    return fig