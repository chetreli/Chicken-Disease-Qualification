import os
import urllib.request as request
from zipfile import ZipFile
from torchvision import models, transforms
import torch.nn as nn
import torch
from pathlib import Path
from ChickenDiseaseClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

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

        self.save_model(
            path=self.config.base_model_path,
            model=self.model
        )

    @staticmethod
    def _prepare_full_model(
        model,
        classes,
        freeze_all,
        freeze_till,
        learning_rate
    ):
        
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False

        elif (freeze_till is not None) and (freeze_till > 0):
            for param in model.features[:-freeze_till].parameters():
                param.requires_grad = False

        num_features = model.classifier.in_features if isinstance(model.classifier, nn.Sequential) else 25088
        #num_features = model.classifier[6].in_features

        model.classifier = nn.Sequential(
            nn.Linear(num_features, classes),
            nn.Softmax(dim=1)
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate
        )

        model.train()

        print(model)

        return model, criterion, optimizer
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(
            path=self.config.base_model_path,
            model=self.model
        )
    
    @staticmethod
    def save_model(path: Path, model: nn.Module):
        torch.save(model.state_dict(), path) 