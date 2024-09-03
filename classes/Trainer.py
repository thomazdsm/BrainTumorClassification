import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Tuple
from sklearn.metrics import precision_score, hamming_loss, mean_squared_error
import numpy as np


class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, n_epochs: int, patience: int) -> None:
        best_val_loss = float("inf")
        patience_counter = 0

        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        train_precisions, val_precisions = [], []
        train_hamming_losses, val_hamming_losses = [], []
        train_rmses, val_rmses = [], []

        for epoch in tqdm(range(n_epochs), desc="Training Epochs"):
            train_loss, train_acc, train_precision, train_hamming, train_rmse = self._train_epoch()
            val_loss, val_acc, val_precision, val_hamming, val_rmse = self._evaluate_epoch()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            train_precisions.append(train_precision)
            val_precisions.append(val_precision)
            train_hamming_losses.append(train_hamming)
            val_hamming_losses.append(val_hamming)
            train_rmses.append(train_rmse)
            val_rmses.append(val_rmse)

            print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Train Precision: {train_precision:.4f} | Val Precision: {val_precision:.4f}")
            print(f"Train Hamming Loss: {train_hamming:.4f} | Val Hamming Loss: {val_hamming:.4f}")
            print(f"Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pt")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        self.model.load_state_dict(torch.load("best_model.pt"))
        self._plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies,
                           train_precisions, val_precisions, train_hamming_losses, val_hamming_losses,
                           train_rmses, val_rmses)

    def _train_epoch(self) -> Tuple[float, float, float, float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        all_labels = []
        all_preds = []

        for inputs, labels in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

        accuracy = correct / total
        precision = precision_score(all_labels, all_preds, average='macro')
        hamming = hamming_loss(all_labels, all_preds)
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))

        return total_loss / len(self.train_loader), accuracy, precision, hamming, rmse

    def _evaluate_epoch(self) -> Tuple[float, float, float, float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        accuracy = correct / total
        precision = precision_score(all_labels, all_preds, average='macro')
        hamming = hamming_loss(all_labels, all_preds)
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))

        return total_loss / len(self.val_loader), accuracy, precision, hamming, rmse

    def _plot_metrics(self, train_losses: List[float], val_losses: List[float],
                      train_accuracies: List[float], val_accuracies: List[float],
                      train_precisions: List[float], val_precisions: List[float],
                      train_hamming_losses: List[float], val_hamming_losses: List[float],
                      train_rmses: List[float], val_rmses: List[float]) -> None:
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss over epochs')

        plt.subplot(2, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy over epochs')

        plt.subplot(2, 2, 3)
        plt.plot(train_precisions, label='Train Precision')
        plt.plot(val_precisions, label='Validation Precision')
        plt.xlabel('Epochs')
        plt.ylabel('Precision')
        plt.legend()
        plt.title('Precision over epochs')

        plt.subplot(2, 2, 4)
        plt.plot(train_hamming_losses, label='Train Hamming Loss')
        plt.plot(val_hamming_losses, label='Validation Hamming Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Hamming Loss')
        plt.legend()
        plt.title('Hamming Loss over epochs')

        plt.tight_layout()
        plt.show()
