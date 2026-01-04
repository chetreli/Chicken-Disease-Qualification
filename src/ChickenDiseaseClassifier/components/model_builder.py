from torchvision import models
import torch.nn as nn

def build_model(arch: dict) -> nn.Module:
    if arch["name"] == "vgg16":
        model = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1
            if arch.get("weights") == "imagenet"
            else None
        )

        if not arch.get("include_top", False):
            model.classifier = nn.Identity()

        model.classifier = nn.Sequential(
            nn.Linear(25088, arch["num_classes"])
        )
        return model

    raise ValueError(f"Unknown architecture: {arch['name']}")
