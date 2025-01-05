import torch
from torch.utils.data import Dataset

class FacialKeypointsDataset(Dataset):
    def __init__(self, images, keypoints=None):
        self.images = torch.tensor(images, dtype=torch.float32).view(-1, 1, 96, 96)
        self.keypoints = torch.tensor(keypoints, dtype=torch.float32) if keypoints is not None else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].unsqueeze(0)

        if self.keypoints is not None:
            return self.images[idx], self.keypoints[idx]
        else:
            return self.images[idx]
