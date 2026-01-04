import os
from pathlib import Path
from torchvision import models, transforms
import torch
import torch.nn as nn
import pytorch_lightning as pl
from ChickenDiseaseClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None

    def get_base_model(self):
        self.model = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1
            if self.config.params_weights == "imagenet"
            else None
        )

        h, w, in_channels = self.config.params_image_size
        
        if not self.config.params_include_top:
            self.model.classifier = nn.Identity()

        self.transform = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406][:in_channels],
                std=[0.229, 0.224, 0.225][:in_channels]
            )
        ])

        self.model.to(self.device)
        self.save_model(self.config.base_model_path)

        return self.model

    @staticmethod
    def _prepare_full_model(
        model,
        classes: int,
        freeze_all=True,
        freeze_till=None,
        learning_rate=0.001):
       
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for param in list(model.features.parameters())[:-freeze_till]:
                param.requires_grad = False

        
        num_features = model.classifier.in_features if isinstance(model.classifier, nn.Sequential) else 25088
        model.classifier = nn.Sequential(
            nn.Linear(num_features, classes),
            nn.Softmax(dim=1)
        )

        model.learning_rate = learning_rate
        model.criterion = nn.CrossEntropyLoss()
        model.train()
        return model

    def update_base_model(self):
        self.model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate)

        self.save_model(self.config.updated_base_model_path)

    def save_model(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
        "state_dict": self.model.state_dict(),
        "arch": {
            "name": "vgg16",
            "weights": self.config.params_weights,
            "include_top": self.config.params_include_top,
            "num_classes": self.config.params_classes
        }
    }, path)



