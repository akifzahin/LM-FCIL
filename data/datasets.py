from torch.utils.data import Subset, DataLoader
class PhaseDataset:
    """Phase dataset that handles dynamic label mapping for federated learning"""

    def __init__(self, parent_dataset, phase_indices, phase_writers=None, round_training_map=None):
        """
        Initialize PhaseDataset with dynamic label mapping support

        Args:
            parent_dataset: The original CustomSubset dataset
            phase_indices: List of indices for this phase (within the parent_dataset)
            phase_writers: List of original writer IDs for this phase (optional)
            round_training_map: Dictionary mapping original writer IDs to new class indices (optional)
        """
        self.parent_dataset = parent_dataset
        self.phase_indices = phase_indices
        self.round_training_map = round_training_map

        # Handle different initialization modes
        if round_training_map is not None:
            # Mode 1: Use provided training map for label remapping
            self.phase_writers = set(round_training_map.keys()) if round_training_map else set()
            print(f"PhaseDataset: Using round_training_map with {len(round_training_map)} writers")
            print(f"  Map: {round_training_map}")
        elif phase_writers is not None:
            # Mode 2: Use phase writers (backward compatibility)
            self.phase_writers = set(phase_writers)
            print(f"PhaseDataset: Using phase_writers: {phase_writers}")
        else:
            # Mode 3: Extract from parent dataset
            self.phase_writers = self._extract_phase_writers()
            print(f"PhaseDataset: Extracted writers: {sorted(self.phase_writers)}")

    def _extract_phase_writers(self):
        """Extract unique writer IDs from the phase indices"""
        writers = set()
        for idx in self.phase_indices:
            ## Get original writer ID through parent dataset
            parent_idx = self.parent_dataset.indices[idx]
            if hasattr(self.parent_dataset.dataset, 'samples'):
                _, original_label = self.parent_dataset.dataset.samples[parent_idx]
            else:
                _, original_label = self.parent_dataset.dataset[parent_idx]
            writers.add(original_label)
        return writers

    def __len__(self):
        return len(self.phase_indices)

    def __getitem__(self, idx):
        """Get item with proper label mapping"""
        # Get the phase index (this is an index into phase_indices)
        parent_dataset_idx = self.phase_indices[idx]

        # Get image and original mapped label from parent dataset
        image, parent_label = self.parent_dataset[parent_dataset_idx]

        # If we have a round training map, we need to remap using original writer IDs
        if self.round_training_map is not None:
            # Get the original writer ID from the parent dataset
            # The parent_dataset_idx is an index into self.parent_dataset
            actual_dataset_idx = self.parent_dataset.indices[parent_dataset_idx]

            # Get original label from the actual dataset
            if hasattr(self.parent_dataset.dataset, 'samples'):
                _, original_writer_id = self.parent_dataset.dataset.samples[actual_dataset_idx]
            else:
                _, original_writer_id = self.parent_dataset.dataset[actual_dataset_idx]

            # Map original writer ID to new class index using round training map
            if original_writer_id in self.round_training_map:
                new_label = self.round_training_map[original_writer_id]

                # Debug for first few samples
                if idx < 3:
                    print(f"  Sample {idx}: Original writer {original_writer_id} -> New label {new_label}")

                return image, new_label
            else:
                print(f"⚠️ Warning: Writer {original_writer_id} not found in round training map")
                return image, parent_label
        else:
            # No remapping needed, use parent dataset's label
            return image, parent_label

    def get_original_labels(self):
        """Get original writer IDs for this phase"""
        return sorted(list(self.phase_writers))


# Additional method for CustomSubset to support PhaseDataset
class CustomSubsetExtended:
    """Extended version of CustomSubset with additional helper methods"""

    def get_original_label_by_index(self, phase_idx):
        """Get original writer ID for a given phase index"""
        # Get the actual dataset index
        dataset_idx = self.indices[phase_idx]

        # Get the original label from the dataset
        if hasattr(self.dataset, 'samples'):
            _, original_label = self.dataset.samples[dataset_idx]
        else:
            _, original_label = self.dataset[dataset_idx]

        return original_label


# Fix for the FederatedClient to use the corrected PhaseDataset
def fix_federated_client_phase_dataset_usage():
    """
    This shows how to fix the FederatedClient.local_train_phase method
    to properly use the updated PhaseDataset class
    """

    # In the FederatedClient.local_train_phase method, replace this section:

    # OLD CODE (around lines 216-222):
    # phase_dataset = PhaseDataset(
    #     parent_dataset=self.dataset,
    #     phase_indices=train_indices,
    #     round_training_map=round_training_map # Pass the complete map
    # )

    # NEW CODE:
    phase_dataset = PhaseDataset(
        parent_dataset=self.dataset,
        phase_indices=train_indices,
        round_training_map=round_training_map  # Now properly supported
    )

    # The rest of the code remains the same
    train_loader = DataLoader(
        phase_dataset,
        batch_size=self.config['client_batch_size'],
        shuffle=True
    )

    return "PhaseDataset usage fixed"


# Also need to add this method to your CustomSubset class
def add_to_custom_subset():
    """
    Add this method to your existing CustomSubset class:
    """

    def get_original_label_by_index(self, phase_idx):
        """Get original writer ID for a given phase index"""
        # Get the actual dataset index
        dataset_idx = self.indices[phase_idx]

        # Get the original label from the dataset
        if hasattr(self.dataset, 'samples'):
            _, original_label = self.dataset.samples[dataset_idx]
        else:
            _, original_label = self.dataset[dataset_idx]

        return original_label




class CustomSubset(Subset):
    """Custom Subset that handles label remapping for federated learning"""
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.label_mapping = {}  # original_label -> model_label
        self.reverse_mapping = {}  # model_label -> original_label
        self.setup_label_mapping()

    def setup_label_mapping(self):
        """Create local label mapping from ImageFolder labels to sequential model labels"""
        # Get all unique labels in this subset
        unique_labels = set()
        for idx in self.indices:
            if hasattr(self.dataset, 'samples'):
                _, label = self.dataset.samples[idx]
            else:
                _, label = self.dataset[idx]
            unique_labels.add(label)

        # Sort labels for consistent mapping
        sorted_labels = sorted(list(unique_labels))

        # Create bidirectional mapping
        for model_idx, original_label in enumerate(sorted_labels):
            self.label_mapping[original_label] = model_idx
            self.reverse_mapping[model_idx] = original_label

        print(f"Label mapping created: {self.label_mapping}")

    def __getitem__(self, idx):
        image, original_label = super().__getitem__(idx)
        # Map original ImageFolder label to sequential model label
        model_label = self.label_mapping.get(original_label, original_label)
        return image, model_label

    def get_original_label(self, model_label):
        """Convert model prediction back to original writer ID"""
        return self.reverse_mapping.get(model_label, model_label)

    def get_original_labels(self):
        """Get all original writer IDs in this subset"""
        return list(self.reverse_mapping.values())

    def update_labels(self, new_mapping):
        """Update label mapping - used for federated coordination"""
        self.label_mapping.update(new_mapping)
        # Update reverse mapping
        self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}


class LabelOffsetDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, label_offset):
        self.base_dataset = base_dataset
        self.label_offset = label_offset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        offset_label = label + self.label_offset
        return image, offset_label

    @property
    def samples(self):
        """Create offset samples list for compatibility"""
        if hasattr(self.base_dataset, 'samples'):
            return [(path, label + self.label_offset) for path, label in self.base_dataset.samples]
        return []

    @property
    def classes(self):
        if hasattr(self.base_dataset, 'classes'):
            return self.base_dataset.classes
        return []


def distribute_writers_to_clients(dataset, num_clients=4, total_clients=4):
    """
    Distribute writers across clients to simulate real forensic scenario
    Each lab (client) has samples from different writers
    """

    # Get all writer classes (0-23)
    all_classes = list(range(24))
    writers_per_client = len(all_classes) // num_clients

    print(f"Distributing {len(all_classes)} writers across {num_clients} clients")
    print(f"Writers per client: {writers_per_client}")

    client_data = {}

    for client_id in range(total_clients):

        start_idx = client_id * writers_per_client
        end_idx = start_idx + writers_per_client

        # Handle last client getting remaining writers
        if client_id == num_clients - 1:
            end_idx = len(all_classes)

        client_writers = all_classes[start_idx:end_idx]
        client_data[client_id] = client_writers

        print(f"Client {client_id}: Writers {client_writers} ({len(client_writers)} writers)")

    return client_data




