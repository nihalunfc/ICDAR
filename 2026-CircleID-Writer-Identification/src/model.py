# --- Model Architecture ---
import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    # Load a pre-trained ResNet34
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    
    # Replace the final layer to match the number of writers in the training set
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model
