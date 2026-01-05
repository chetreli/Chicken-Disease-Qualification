import os
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ChickenDiseaseClassifier.entity.config_entity import PrepareBaseModelConfig
from ChickenDiseaseClassifier.entity.config_entity import TrainingConfig
from ChickenDiseaseClassifier.components.model_builder import build_model

class Training:
    def __init__(self, config: TrainingConfig, config_model: PrepareBaseModelConfig):
        self.config = config
        self.config_model = config_model

    def get_base_model(self):
        checkpoint = torch.load(self.config.updated_base_model_path)
        
        self.model = build_model(checkpoint["arch"])
        self.model.load_state_dict(checkpoint["state_dict"])
    def train_valid_dataloader(self):
        h, w = self.config.params_image_size[:2]

        base_transform = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        if self.config.params_is_augmentation:
            train_transform = transforms.Compose([
                transforms.Resize((h, w)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(40),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            train_transform = base_transform

        full_dataset = datasets.ImageFolder(
            root=self.config.training_data,
            transform=train_transform
        )

        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size

        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.config.params_batch_size,
            shuffle=True,
            num_workers=4
        )

        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.config.params_batch_size,
            shuffle=False,
            num_workers=4
        )

    @staticmethod
    def save_model(path: Path, model: torch.nn.Module):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)


    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
    
    
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config_model.params_learning_rate
        )
    
        num_epochs = self.config.params_epochs
        num_classes = 2

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for x, y in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]"):
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                outputs = self.model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * x.size(0)

                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            train_loss = running_loss / total
            train_acc = correct / total

            print(
                f"Epoch {epoch+1}/{num_epochs} "
                f"TRAIN loss: {train_loss:.4f}, acc: {train_acc:.4f}"
            )

            
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            
            tp = torch.zeros(num_classes, device=device)
            fp = torch.zeros(num_classes, device=device)
            fn = torch.zeros(num_classes, device=device)

            with torch.no_grad():
                for x, y in self.val_loader:
                    x, y = x.to(device), y.to(device)

                    outputs = self.model(x)
                    loss = criterion(outputs, y)
                    val_loss += loss.item() * x.size(0)

                    preds = outputs.argmax(dim=1)

                    correct += (preds == y).sum().item()
                    total += y.size(0)

                    for c in range(num_classes):
                        tp[c] += ((preds == c) & (y == c)).sum()
                        fp[c] += ((preds == c) & (y != c)).sum()
                        fn[c] += ((preds != c) & (y == c)).sum()

            val_loss /= total
            val_acc = correct / total

            precision = (tp / (tp + fp + 1e-8)).mean().item()
            recall = (tp / (tp + fn + 1e-8)).mean().item()
            f1 = (2 * precision * recall) / (precision + recall + 1e-8)

            print(
                f"Epoch {epoch+1}/{num_epochs} "
                f"VAL loss: {val_loss:.4f}, "
                f"acc: {val_acc:.4f}, "
                f"precision: {precision:.4f}, "
                f"recall: {recall:.4f}, "
                f"f1: {f1:.4f}"
            )

        
        path = self.config.trained_model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "arch": {
                "name": "vgg16",
                "weights": self.config_model.params_weights,
                "include_top": self.config_model.params_include_top,
                "num_classes": self.config_model.params_classes
            }
        }, path)

        print(f"Model saved at {path}")