import os
from pathlib import Path

import albumentations as A
import cv2
import torch
import torch.utils.data as data
from albumentations.pytorch import ToTensorV2


class PolypDataset(data.Dataset):
    """Binary polyp segmentation dataset with basename-safe image/mask pairing."""

    def __init__(self, image_root, gt_root, trainsize, augmentation=False,
                 split="train", color_image=True):
        self.trainsize = int(trainsize)
        self.color_image = bool(color_image)
        self.augmentation = bool(augmentation)
        self.split = split

        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        image_map = {
            Path(f).stem: os.path.join(image_root, f)
            for f in os.listdir(image_root)
            if Path(f).suffix.lower() in exts
        }
        mask_map = {
            Path(f).stem: os.path.join(gt_root, f)
            for f in os.listdir(gt_root)
            if Path(f).suffix.lower() in exts
        }
        common = sorted(set(image_map).intersection(mask_map))
        if not common:
            raise RuntimeError(f"No paired images/masks found in {image_root} and {gt_root}")

        self.samples = [(image_map[k], mask_map[k]) for k in common]

        if self.color_image:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        else:
            mean, std = [0.5], [0.229]

        transforms = []
        if self.split == "train" and self.augmentation:
            transforms.extend([
                A.Rotate(limit=90, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
            ])
        transforms.extend([
            A.Resize(self.trainsize, self.trainsize),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
        self.transform = A.Compose(transforms)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]

        if self.color_image:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            image = image[..., None]

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")

        original_shape = tuple(mask.shape[:2])  # (H, W) before resize
        augmented = self.transform(image=image, mask=mask)
        image_t = augmented["image"]
        mask_t = augmented["mask"]

        # Robust binary conversion for 0/255 and 0/1-style masks.
        if mask_t.max() > 127:
            mask_t = (mask_t > 20).long()
        else:
            mask_t = (mask_t >= 1).long()
        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0)

        if self.split == "train":
            return image_t, mask_t

        name = Path(image_path).stem + ".png"
        return image_t, mask_t, original_shape, name


def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=False,
               num_workers=4, pin_memory=True, augmentation=False,
               split="train", color_image=True):
    dataset = PolypDataset(
        image_root=image_root,
        gt_root=gt_root,
        trainsize=trainsize,
        augmentation=augmentation,
        split=split,
        color_image=color_image,
    )
    return data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
    )
