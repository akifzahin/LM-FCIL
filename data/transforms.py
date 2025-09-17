# Data transformations
from torchvision import transforms

def get_train_transform(image_size: int = 64):
    """Return training data augmentation pipeline."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_val_transform(image_size: int = 64):
    """Return validation/test preprocessing pipeline."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])




