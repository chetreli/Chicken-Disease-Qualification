import numpy as np
import torch
from torchvision import transforms
import os
from PIL import Image

from ChickenDiseaseClassifier.components.model_builder import build_model

class PredictionPipeline:
    def __init__(self, filename): 
        self.filename = filename
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self):
        checkpoint = torch.load(os.path.join("artifacts", "training", "model.pth"))
        model = build_model(checkpoint["arch"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        imagename = self.filename
        image = Image.open(imagename).convert("RGB")
        image = transform(image)         
        image = image.unsqueeze(0)

        image = image.to(self.device)
        model = model.to(self.device)

        with torch.no_grad():
            outputs = model(image)
            print(image)
            print(outputs)
            print(torch.argmax(outputs, dim=1))
            prediction = torch.argmax(outputs, dim=1).item()

        

        if prediction == 1:
            prediction = 'Healthy'
            return [{ "image" : prediction}]
        else:
            prediction = 'Coccidiosis'
            return [{ "image" : prediction}]


