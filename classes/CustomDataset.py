import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import List, Tuple

# Definição do dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, root_dir: str, transform: transforms.Compose, split: str = 'training'):
        self.root_dir = os.path.join(root_dir, split)
        self.categories = os.listdir(self.root_dir)
        self.transform = transform
        self.split = split
        self.int2id = {i: category for i, category in enumerate(self.categories)}
        self.data = self._load_data()

    def _load_data(self) -> List[Tuple[str, int]]:
        data = []
        for i, category in enumerate(self.categories):
            category_path = os.path.join(self.root_dir, category)
            for file_name in os.listdir(category_path):
                img_path = os.path.join(category_path, file_name)
                data.append((img_path, i))
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("L")
        label = torch.tensor(label, dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image.to(device), label.to(device)
