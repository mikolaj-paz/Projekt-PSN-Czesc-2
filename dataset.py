import torch
from torch.utils.data import Dataset
from augmentation import combined_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Model na GPU (jeśli dostępne)

class FacialKeypointsDataset(Dataset):
    def __init__(self, images, keypoints, transform_image, transform_combined=None):
        self.images = images
        self.keypoints = keypoints
        self.transform_image = transform_image
        self.transform_combined = transform_combined

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        keypoints_orig = self.keypoints[idx]
        keypoints = keypoints_orig.copy()

        # Nałóż transformacje obrazu
        if self.transform_image:
            image = self.transform_image(image)
        
        # Nałóż transformacje łączone
        orig_image = image.detach().clone()
        if self.transform_combined:
            comb_transform = combined_transform()
            image, keypoints = comb_transform(image, keypoints)
        
        if keypoints is not None:
            return image, torch.from_numpy(keypoints).float().to(device)
        else:
            return orig_image, torch.from_numpy(keypoints_orig).float().to(device)
