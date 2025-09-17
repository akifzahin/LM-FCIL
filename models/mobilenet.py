import torch
import torch.nn as nn
from torchvision import models

class WriterIdentificationMobileModel(nn.Module):
    def __init__(self, num_classes: int = 24):
        """
        num_classes:   initial number of writers
        max_classes:   upper bound on total writers you'll ever support
        """
        super().__init__()
        # Backbone - using MobileNetV3 Small for <10MB model size
        self.backbone = models.mobilenet_v3_small(weights='DEFAULT')
        in_feats = self.backbone.classifier[3].in_features

        # Replace the classifier with our custom head
        self.backbone.classifier = nn.Sequential(
            self.backbone.classifier[0],  # AdaptiveAvgPool2d
            self.backbone.classifier[1],  # Flatten
            self.backbone.classifier[2],  # Dropout
            nn.Linear(in_feats, num_classes)  # Our custom final layer
        )

        self.current_classes = num_classes
        # Initialize only the active slice [0:num_classes)
        # nn.init.normal_(self.backbone.classifier[3].weight[:num_classes], mean=0.0, std=0.01)
        # nn.init.zeros_(self.backbone.classifier[3].bias[:num_classes])

    @property
    def classifier(self) -> nn.Linear:
        """Expose the full (pre‑allocated) head module for compatibility."""
        return self.backbone.classifier[3]  # The Linear layer in the Sequential

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns logits only for the active classes:
           shape = [B, current_classes]
        """
        logits = self.backbone(x)              # [B, max_classes]
        # return logits[:, : self.current_classes]
        return logits

    def expand_classifier(self, new_num_classes):
        """Expand the classifier to handle more classes"""
        current_linear_layer = self.backbone.classifier[3]
        current_num_classes = current_linear_layer.out_features

        # Don't expand if already sufficient
        if new_num_classes <= current_num_classes:
            return

        print(f"Expanding classifier from {current_num_classes} to {new_num_classes} classes")

        # Store old weights and bias
        old_weights = current_linear_layer.weight.data.clone()
        old_bias = current_linear_layer.bias.data.clone()

        # Get device and dtype from current layer
        device = old_weights.device
        dtype = old_weights.dtype

        # Create new classifier with expanded size
        num_features = current_linear_layer.in_features
        new_linear_layer = nn.Linear(num_features, new_num_classes)

        # Move new layer to the same device
        new_linear_layer = new_linear_layer.to(device)

        # Initialize new weights with small random values
        torch.nn.init.normal_(new_linear_layer.weight.data, mean=0.0, std=0.01)
        torch.nn.init.constant_(new_linear_layer.bias.data, 0.0)

        # Copy old weights and biases to preserve learned representations
        new_linear_layer.weight.data[:current_num_classes] = old_weights
        new_linear_layer.bias.data[:current_num_classes] = old_bias

        # Replace the linear layer in the Sequential classifier
        self.backbone.classifier[3] = new_linear_layer

        print(f"✅ Classifier expanded successfully. New shape: {self.backbone.classifier[3].weight.shape}")

