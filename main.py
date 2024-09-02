import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from torchvision import transforms

from classes.CustomDataset import CustomDataset
from classes.TumorClassifier import TumorClassifier
from classes.Trainer import Trainer
from classes.CrossValidator import CrossValidator

# Definição do dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    ROOT_DIR = '/home/thomaz/Projects/EngComp/BrainTumorClassification/datasets/masoud_nickparvar'

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=0.25),
        transforms.RandomAffine(5, translate=(0.01, 0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = CustomDataset(root_dir=ROOT_DIR, transform=train_transform, split='training')
    test_dataset = CustomDataset(root_dir=ROOT_DIR, transform=test_transform, split='testing')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    model = TumorClassifier(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)
    trainer.train(n_epochs=20, patience=8)

    cross_validator = CrossValidator(train_dataset, TumorClassifier, 4, criterion, torch.optim.Adam, 20, 8, k=5)
    cross_validator.run()


if __name__ == "__main__":
    main()
