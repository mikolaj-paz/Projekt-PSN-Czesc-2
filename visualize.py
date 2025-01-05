import torch
import matplotlib.pyplot as plt

def visualize_predictions(model, device, test_loader, num_images=5):
    model.eval()
    with torch.no_grad():
        # Pobierz paczkę danych z test_loader
        batch = next(iter(test_loader))
        
        # Jeśli batch zawiera obrazy i etykiety
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:  # Jeśli batch zawiera tylko obrazy
            images = batch
        
        # Przekształć listę na tensor, jeśli to konieczne
        if isinstance(images, list):
            images = torch.stack(images)

        images = images.to(device)
        predictions = model(images).cpu().numpy()

        for i in range(num_images):
            img = images[i].cpu().numpy().squeeze()
            pred = predictions[i].reshape(-1, 2) # (x, y)

            plt.imshow(img, cmap='gray')
            plt.scatter(pred[:, 0], pred[:, 1], c='r', s=10) # Skala do wymiarów obrazu
            plt.show()
    