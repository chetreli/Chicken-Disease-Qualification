import torch
import torch.nn as nn
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from ChickenDiseaseClassifier.components.model_builder import build_model
from ChickenDiseaseClassifier.entity.config_entity import EvaluationConfig


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        checkpoint = torch.load(self.config.path_of_model, map_location=self.device)

        self.model = build_model(checkpoint["arch"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def valid_dataloader(self):
        h, w = self.config.params_image_size[:2]

        transform = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        dataset = datasets.ImageFolder(
            root=self.config.training_data,
            transform=transform
        )

        self.loader = DataLoader(
            dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False,
            num_workers=0  
        )

    def evaluate(self):
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in tqdm(self.loader, desc="Evaluating"):
                x, y = x.to(self.device), y.to(self.device)

                outputs = self.model(x)
                loss = criterion(outputs, y)

                total_loss += loss.item() * x.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        self.loss = total_loss / total
        self.accuracy = correct / total

        print(f"Evaluation loss: {self.loss:.4f}")
        print(f"Evaluation accuracy: {self.accuracy:.4f}")

    def save_score(self):
        scores = {
            "loss": self.loss,
            "accuracy": self.accuracy
        }

        path = Path(self.config.score_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        import json
        with open(path, "w") as f:
            json.dump(scores, f, indent=4)
