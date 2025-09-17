from data import get_train_transform, get_val_transform

from data. import CustomSubset, PhaseDataset, CustomSubsetExtended, LabelOffsetDataset, distribute_writers_to_clients, add_to_custom_subset, fix_federated_client_phase_dataset_usage
from data.splits import create_train_test_split, create_client_datasets
from data.sampler import create_balanced_sampler


train_transform = get_train_transform(image_size=64)
val_transform = get_val_transform(image_size=64)

train_dataset = CustomSubset(root="data/train", transform=train_transform)
val_dataset = CustomSubset(root="data/val", transform=val_transform)
full_dataset = CustomSubset(root="data/full", transform=val_transform)