import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Ham10000Dataset(Dataset):
    """HAM10000 dataset loader.

    Expects a folder layout:
      data/
        images/
          ISIC_0000000.jpg
          ...
        HAM10000_metadata.csv

    The CSV should contain at least columns: `image_id`, `dx` (diagnosis label).
    """

    def __init__(
        self, images_root, metadata_csv, transform=None, target_transform=None
    ):
        self.images_root = images_root
        self.metadata = pd.read_csv(metadata_csv)
        self.transform = transform
        self.target_transform = target_transform

        if "image_id" not in self.metadata.columns:
            raise ValueError("metadata CSV must contain 'image_id' column")
        if "dx" not in self.metadata.columns:
            raise ValueError("metadata CSV must contain 'dx' column")

        # image filenames are usually image_id + '.jpg'
        self.metadata["image_path"] = self.metadata["image_id"].astype(str) + ".jpg"

        self.samples = self.metadata[["image_path", "dx"]].values.tolist()

        # build label -> index mapping
        self.label_names = sorted(self.metadata["dx"].unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.label_names)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label_name = self.samples[idx]
        img_path = os.path.join(self.images_root, img_name)
        if not os.path.exists(img_path):
            # fallback: maybe file extension is PNG
            alt = os.path.splitext(img_path)[0] + ".png"
            if os.path.exists(alt):
                img_path = alt
            else:
                raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        label = self.class_to_idx[label_name]
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label


def get_transforms(image_size=224, is_train=True):
    if is_train:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    return transform
