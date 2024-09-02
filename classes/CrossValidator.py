import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from classes.Trainer import Trainer

# Definição do dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CrossValidator:
    def __init__(self, dataset: Dataset, model_cls: nn.Module, num_classes: int,
                 criterion: nn.Module, optimizer_cls: torch.optim.Optimizer,
                 n_epochs: int, patience: int, k: int = 5):
        self.dataset = dataset
        self.model_cls = model_cls
        self.num_classes = num_classes
        self.criterion = criterion
        self.optimizer_cls = optimizer_cls
        self.n_epochs = n_epochs
        self.patience = patience
        self.k = k

    def run(self) -> None:
        dataset_size = len(self.dataset)
        fold_size = dataset_size // self.k
        indices = list(range(dataset_size))
        random.shuffle(indices)

        for i in range(self.k):
            val_indices = indices[i * fold_size:(i + 1) * fold_size]
            train_indices = indices[:i * fold_size] + indices[(i + 1) * fold_size:]

            train_subset = torch.utils.data.Subset(self.dataset, train_indices)
            val_subset = torch.utils.data.Subset(self.dataset, val_indices)

            train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=4)

            model = self.model_cls(self.num_classes).to(device)
            optimizer = self.optimizer_cls(model.parameters(), lr=0.001)

            print(f"Fold {i + 1}/{self.k}")
            trainer = Trainer(model, train_loader, val_loader, self.criterion, optimizer, device)
            trainer.train(self.n_epochs, self.patience)
