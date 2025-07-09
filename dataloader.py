import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

# === Set working directory if needed ===
os.chdir('data')  # Adjust or remove if not required

# === Dataset Loading Functions ===
def load_dataset_from_dir(path, exclude_name='AFLW2000.npz'):
    dataset_list, labels_list = [], []
    for file in os.listdir(path):
        if file.endswith('.npz') and file != exclude_name:
            data = np.load(os.path.join(path, file))
            images, poses = data['image'], data['pose']
            valid = (poses <= 99.0).all(axis=1) & (poses >= -99.0).all(axis=1)
            dataset_list.append(images[valid])
            labels_list.append(poses[valid])
    return np.concatenate(dataset_list, axis=0), np.concatenate(labels_list, axis=0)

def load_single_dataset(path):
    data = np.load(path)
    images, poses = data['image'], data['pose']
    valid = (poses <= 99.0).all(axis=1) & (poses >= -99.0).all(axis=1)
    return images[valid], poses[valid]

# === Custom Augmentations ===
def random_crop(img, dn):
    dx, dy = np.random.randint(0, dn, 2)
    h, w = img.shape[:2]
    cropped = img[dy:h-(dn-dy), dx:w-(dn-dx), :]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)

def random_crop_black(img, dn):
    dx, dy = np.random.randint(0, dn, 2)
    dx_shift, dy_shift = np.random.randint(0, dn, 2)
    h, w = img.shape[:2]
    out = np.zeros_like(img)
    cropped = img[dy:h-(dn-dy), dx:w-(dn-dx), :]
    out[dy_shift:h-(dn-dy_shift), dx_shift:w-(dn-dx_shift), :] = cropped
    return out

def random_crop_white(img, dn):
    dx, dy = np.random.randint(0, dn, 2)
    dx_shift, dy_shift = np.random.randint(0, dn, 2)
    h, w = img.shape[:2]
    out = np.ones_like(img) * 255
    cropped = img[dy:h-(dn-dy), dx:w-(dn-dx), :]
    out[dy_shift:h-(dn-dy_shift), dx_shift:w-(dn-dx_shift), :] = cropped
    return out

def augment_image(img):
    rand_r = np.random.rand()
    dn = np.random.randint(1, 16)
    if rand_r < 0.25:
        img = random_crop(img, dn)
    elif rand_r < 0.5:
        img = random_crop_black(img, dn)
    elif rand_r < 0.75:
        img = random_crop_white(img, dn)

    if np.random.rand() > 0.3:
        zoom_factor = np.random.uniform(0.8, 1.2)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), 0, zoom_factor)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return img

# === PyTorch Dataset ===
class HeadPoseDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images = images.astype(np.uint8)
        self.labels = labels.astype(np.float32)
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.augment:
            img = augment_image(img)
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

# === Load Datasets ===
X_train, Y_train = load_dataset_from_dir('type1')
X_test_1, Y_test_1 = load_single_dataset('BIWI_noTrack.npz')
X_test_2, Y_test_2 = load_single_dataset(os.path.join('type1', 'AFLW2000.npz'))

print("Train shape:", X_train.shape, Y_train.shape)
print("BIWI shape:", X_test_1.shape, Y_test_1.shape)
print("AFLW shape:", X_test_2.shape, Y_test_2.shape)

# === Label Statistics ===
def print_stats(name, values):
    print(f"{name}: min={values.min():.2f}, max={values.max():.2f}, mean={values.mean():.2f}, std={values.std():.2f}")

print_stats("Yaw", Y_train[:, 0])
print_stats("Pitch", Y_train[:, 1])
print_stats("Roll", Y_train[:, 2])

# === Example Usage ===
# train_dataset = HeadPoseDataset(X_train, Y_train, augment=True)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
