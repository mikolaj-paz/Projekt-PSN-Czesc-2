import torch
import torchvision as tv
import torchvision.transforms.functional
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Model na GPU (jeśli dostępne)
img_size = 96

class to_tensor(object):
    def __call__(self, image):
        return torch.from_numpy(image)

class add_noise(object):
    def __init__(self, mean=0., std=.01, p=.25):
        self.std = std
        self.mean = mean
        self.p = p

    def __call__(self, tensor):
        if np.random.uniform() < self.p:
            return tensor + torch.randn(tensor.size()).to(device) * self.std + self.mean
        else:
            return tensor
        
class add_blur(object):
    def __init__(self, kernel_size=(7, 7), sigma=(.01, 1.5), p=.25):
        self.transform = tv.transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        self.p = p
    
    def __call__(self, img):
        if np.random.uniform() < self.p:
            img = self.transform(img)
        return img
    
class adjust_contrast(object):
    def __init__(self, contrast_range=(0.5, 1.5), p=0.25):
        self.contrast_range = contrast_range
        self.p = p

    def __call__(self, tensor):
        if np.random.uniform() < self.p:
            contrast_factor = np.random.uniform(*self.contrast_range)

            # Zmień kontrast
            mean = tensor.mean()
            contrast_tensor = (tensor - mean) * contrast_factor + mean

            return contrast_tensor
        else:
            return tensor
        
def transform_keypoints_rotate(keypoints, angles):
    # Obróć punkty charakterystyczne
    newpred = np.zeros(keypoints.shape)
    rads = np.radians(angles)
    newpred[0::2] = (keypoints[0::2] - img_size/2)*np.cos(rads) + (keypoints[1::2] - img_size/2)*np.sin(rads) + img_size/2
    newpred[1::2] = -(keypoints[0::2] - img_size/2)*np.sin(rads) + (keypoints[1::2] - img_size/2)*np.cos(rads) + img_size/2
    return newpred

def transform_keypoints_shift(keypoints, shift_x, shift_y):
    # Przesun punkty charakterystyczne
    newpred = np.zeros(keypoints.shape)
    newpred[0::2] = keypoints[0::2] + shift_x
    newpred[1::2] = keypoints[1::2] + shift_y
    return newpred

def transform_keypoints_scale(keypoints, factor):
    # Zmien skale punktow charakterystycznych
    newpred = np.zeros(keypoints.shape)
    newpred[0::2] = (keypoints[0::2] - img_size/2) * factor + img_size/2
    newpred[1::2] = (keypoints[1::2] - img_size/2) * factor + img_size/2
    return newpred

# Transformacje obrazów walidacyjnych    
transform_val_image = tv.transforms.Compose([
    to_tensor(),
    tv.transforms.Normalize(mean=[.5], std=[.5]),
])

# Transformacje obrazów treningowych
transform_train_image = tv.transforms.Compose([
    to_tensor(),
    tv.transforms.Normalize(mean=[.5], std=[.5]),
    add_noise(),
    add_blur(),
    adjust_contrast(),
])

class combined_transform(object):
    def __init__(self):
        self.range_angle = 20
        self.shift_pixel = img_size // 12
        self.scale_min = .8
        self.scale_max = 1.2
        self.p_aff = .7
        self.p_rot = .3

    def __call__(self, image, keypoints):
        # Rotacja
        if np.random.uniform() < self.p_rot:
            angles = np.random.uniform(-self.range_angle/2, self.range_angle/2)
        else:
            angles = 0

        # Skala
        if np.random.uniform() < self.p_aff:
            factors = np.random.uniform(self.scale_max, self.scale_min)
        else:
            factors = 1
        
        # Przesuniecie
        if np.random.uniform() < self.p_aff:
            shifts_x = np.random.uniform(-self.shift_pixel, self.shift_pixel)
            shifts_y = np.random.uniform(-self.shift_pixel, self.shift_pixel)
        else:
            shifts_x = 0
            shifts_y = 0

        # Nałóż transformacje na obraz
        image = torchvision.transforms.functional.affine(
            image,
            angle=-angles,
            scale=factors,
            translate=(shifts_x, shifts_y),
            shear=0,
        )

        # Nałóż transformacje na punkty charakterystyczne
        keypoints = transform_keypoints_rotate(keypoints, angles)
        keypoints = transform_keypoints_scale(keypoints, factors)
        keypoints = transform_keypoints_shift(keypoints, shifts_x, shifts_y)

        if any(keypoint < 0 or keypoint > img_size for keypoint in keypoints):
            return image, None

        return image, keypoints