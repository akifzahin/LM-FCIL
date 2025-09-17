import torch
import torch.nn as nn
from torchvision import models

class WriterIdentificationEfficientModel(nn.Module):
    def __init__(self, num_classes=24):
        super(WriterIdentificationEfficientModel, self).__init__()
        # Load pretrained EfficientNet-B1
        self.backbone = models.efficientnet_b1(weights='DEFAULT')

        # EfficientNet-B1 classifier is a Sequential with Dropout and Linear
        # Get the input features from the last layer
        num_features = self.backbone.classifier[1].in_features

        # Replace the classifier with a simple Linear layer
        self.backbone.classifier = nn.Linear(num_features, num_classes)
        self.current_classes = self.backbone.classifier.out_features

    @property
    def classifier(self):
        """Property to expose backbone.classifier as classifier for compatibility"""
        return self.backbone.classifier

    @classifier.setter
    def classifier(self, value):
        """Setter to allow assignment to classifier"""
        self.backbone.classifier = value

    def forward(self, x):
        return self.backbone(x)

    def expand_classifier(self, new_num_classes):
        """Expand the classifier to handle more classes"""
        current_num_classes = self.backbone.classifier.out_features

        # Don't expand if already sufficient
        if new_num_classes <= current_num_classes:
            return

        print(f"Expanding classifier from {current_num_classes} to {new_num_classes} classes")

        # Store old weights and bias
        old_weights = self.backbone.classifier.weight.data.clone()
        old_bias = self.backbone.classifier.bias.data.clone()

        # Get device and dtype from current layer
        device = old_weights.device
        dtype = old_weights.dtype

        # Create new classifier with expanded size
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(num_features, new_num_classes)

        # Move new layer to the same device
        self.backbone.classifier = self.backbone.classifier.to(device)

        # Initialize new weights with small random values
        torch.nn.init.normal_(self.backbone.classifier.weight.data, mean=0.0, std=0.01)
        torch.nn.init.constant_(self.backbone.classifier.bias.data, 0.0)

        # Copy old weights and biases to preserve learned representations
        self.backbone.classifier.weight.data[:current_num_classes] = old_weights
        self.backbone.classifier.bias.data[:current_num_classes] = old_bias

        print(f"✅ Classifier expanded successfully. New shape: {self.backbone.classifier.weight.shape}")