import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class PolypDataset(Dataset):
    """
    Dataset for polyp segmentation with optional augmentation.
    """

    def __init__(self, image_root, mask_root, trainsize, augmentations=False):
        self.trainsize = trainsize
        self.augmentations = augmentations

        self.images = sorted(
            [
                os.path.join(image_root, f)
                for f in os.listdir(image_root)
                if f.endswith((".jpg", ".png"))
            ]
        )
        self.masks = sorted(
            [
                os.path.join(mask_root, f)
                for f in os.listdir(mask_root)
                if f.endswith(".png")
            ]
        )

        self._filter_valid_files()
        self.size = len(self.images)
        self._init_transforms()

    def _filter_valid_files(self):
        self.images, self.masks = zip(
            *[
                (img_path, mask_path)
                for img_path, mask_path in zip(self.images, self.masks)
                if Image.open(img_path).size == Image.open(mask_path).size
            ]
        )

    def _init_transforms(self):
        base_transforms = [
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
        ]

        if self.augmentations:
            aug_transforms = [
                transforms.RandomRotation(90),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
            self.img_transform = transforms.Compose(
                aug_transforms
                + base_transforms
                + [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            )
            self.mask_transform = transforms.Compose(aug_transforms + base_transforms)
        else:
            self.img_transform = transforms.Compose(
                base_transforms
                + [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            )
            self.mask_transform = transforms.Compose(base_transforms)

    def __getitem__(self, index):
        image = self._load_image(self.images[index])
        print(f"Image size: {image.size}")
        mask = self._load_mask(self.masks[index])

        seed = 1234
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.img_transform(image)

        random.seed(seed)
        torch.manual_seed(seed)
        mask = self.mask_transform(mask)

        return image, mask

    def _load_image(self, path):
        with open(path, "rb") as f:
            return Image.open(f).convert("RGB")

    def _load_mask(self, path):
        with open(path, "rb") as f:
            return Image.open(f).convert("L")

    def __len__(self):
        return self.size


def get_loader(
    image_root,
    mask_root,
    batchsize,
    trainsize,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    augmentation=False,
):
    dataset = PolypDataset(image_root, mask_root, trainsize, augmentation)
    return DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


class TestDataset:
    def __init__(self, image_root, mask_root, testsize):
        self.testsize = testsize
        self.images = sorted(
            [
                os.path.join(image_root, f)
                for f in os.listdir(image_root)
                if f.endswith((".jpg", ".png"))
            ]
        )
        self.masks = sorted(
            [
                os.path.join(mask_root, f)
                for f in os.listdir(mask_root)
                if f.endswith((".tif", ".png"))
            ]
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize((testsize, testsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.mask_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        if self.index >= self.size:
            self.index = 0  # reset khi hết dataset
        image = self._load_image(self.images[self.index])
        mask = self._load_mask(self.masks[self.index])
        name = os.path.basename(self.images[self.index]).replace(".jpg", ".png")
        self.index += 1
        return self.transform(image).unsqueeze(0), mask, name

    def _load_image(self, path):
        with open(path, "rb") as f:
            return Image.open(f).convert("RGB")

    def _load_mask(self, path):
        with open(path, "rb") as f:
            return Image.open(f).convert("L")


# import glob

# class TestDataset(Dataset):
#     def __init__(self, image_root, mask_root, testsize, transform=None, mask_transform=None):
#         self.image_files = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.lower().endswith(('.jpg','.png'))])
#         self.mask_files = sorted([os.path.join(mask_root, f) for f in os.listdir(mask_root) if f.lower().endswith(('.tif','.png','.jpg'))])
#         assert len(self.image_files) == len(self.mask_files), f"image/mask count mismatch: {len(self.image_files)} vs {len(self.mask_files)}"
#         self.testsize = testsize
#         self.transform = transform or transforms.Compose([
#             transforms.Resize((testsize, testsize)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
#         ])
#         self.mask_transform = mask_transform or transforms.Compose([
#             transforms.Resize((testsize, testsize)),
#             transforms.ToTensor()
#         ])

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img = Image.open(self.image_files[idx]).convert('RGB')
#         mask = Image.open(self.mask_files[idx]).convert('L')
#         img_t = self.transform(img)
#         mask_t = self.mask_transform(mask)  # tensor shape [1,H,W]
#         # return: image tensor, mask numpy (H,W) or mask tensor => adapt evaluate accordingly
#         return img_t, mask_t.squeeze(0).numpy(), os.path.basename(self.image_files[idx])
