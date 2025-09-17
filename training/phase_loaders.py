import torch
import numpy as np
from torch.utils.data import DataLoader

from data.datasets import CustomSubset
from federated.client import FederatedClient2


def initialize_clients(client_datasets, writers_per_phase, FL_CONFIG, model_class):
    """
    Initialize federated clients with their respective datasets.
    """
    template_model = model_class(num_classes=0)
    clients = {}

    for client_id in range(1, FL_CONFIG["num_clients"]):
        FL_CONFIG["writers_per_phase"] = writers_per_phase[client_id]
        client_dataset = client_datasets[client_id]
        clients[client_id] = FederatedClient2(
            client_id, client_dataset, template_model, FL_CONFIG, start_phase=1
        )

    print(f"Created {len(clients)} federated clients")
    return clients


def build_cumulative_label_mapping(all_clients, combined_test_dataset, FL_CONFIG):
    """
    Builds a mapping of original writer IDs -> new labels across all phases.
    """
    print("Extracting original writer IDs from global test dataset...")
    global_test_original_writers = [
        label.item() if torch.is_tensor(label) else label
        for _, label in combined_test_dataset
    ]
    global_test_original_writers = np.array(global_test_original_writers)

    all_discovered_writers = set()
    cumulative_label_mapping = {}
    next_available_label = 0

    for phase in range(1, FL_CONFIG["phases"] + 1):
        print(f"\n=== PHASE {phase} ===")
        phase_new_writers = set()

        for client_id, client in all_clients.items():
            if hasattr(client, "phase_datasets") and phase in client.phase_datasets:
                phase_dataset = client.phase_datasets[phase]
                client_phase_writers = (
                    set(phase_dataset.get_original_labels())
                    if hasattr(phase_dataset, "get_original_labels")
                    else set(
                        phase_dataset.phase_writers
                        if hasattr(phase_dataset, "phase_writers")
                        else [
                            label.item() if torch.is_tensor(label) else label
                            for _, label in phase_dataset
                        ]
                    )
                )

                new_writers_for_client = client_phase_writers - all_discovered_writers
                phase_new_writers.update(new_writers_for_client)

                if new_writers_for_client:
                    print(f"Client {client_id}: new writers {sorted(new_writers_for_client)}")

        sorted_new_writers = sorted(list(phase_new_writers))
        for original_writer_id in sorted_new_writers:
            cumulative_label_mapping[original_writer_id] = next_available_label
            next_available_label += 1

        all_discovered_writers.update(phase_new_writers)

        print(f"Phase {phase}: {len(phase_new_writers)} new writers: {sorted_new_writers}")
        print(
            f"New label assignments: "
            f"{[(w, cumulative_label_mapping[w]) for w in sorted_new_writers]}"
        )
        print(f"Total writers so far: {len(all_discovered_writers)}, next label: {next_available_label}")

    return cumulative_label_mapping, global_test_original_writers


def create_phase_test_loaders(all_clients, combined_test_dataset, cumulative_label_mapping, FL_CONFIG, worker_init_fn):
    """
    Create test DataLoaders for each phase using cumulative discovered writers.
    """
    phase_global_test_loaders = {}
    cumulative_discovered_writers = set()

    for phase in range(1, FL_CONFIG["phases"] + 1):
        print(f"\nPhase {phase}:")
        phase_new_writers = set()

        for client_id, client in all_clients.items():
            if hasattr(client, "phase_datasets") and phase in client.phase_datasets:
                phase_dataset = client.phase_datasets[phase]
                client_phase_writers = (
                    set(phase_dataset.get_original_labels())
                    if hasattr(phase_dataset, "get_original_labels")
                    else set(
                        phase_dataset.phase_writers
                        if hasattr(phase_dataset, "phase_writers")
                        else [
                            label.item() if torch.is_tensor(label) else label
                            for _, label in phase_dataset
                        ]
                    )
                )
                phase_new_writers.update(client_phase_writers)

        cumulative_discovered_writers.update(phase_new_writers)

        phase_label_mapping = {
            original_id: cumulative_label_mapping[original_id]
            for original_id in cumulative_discovered_writers
        }

        mask = np.isin(global_test_original_writers, list(cumulative_discovered_writers))
        phase_test_indices = np.where(mask)[0].tolist()

        phase_test_subset = CustomSubset(combined_test_dataset, phase_test_indices)
        phase_test_subset.label_mapping = phase_label_mapping
        phase_test_subset.reverse_mapping = {v: k for k, v in phase_label_mapping.items()}

        phase_global_test_loaders[phase] = DataLoader(
            phase_test_subset,
            batch_size=FL_CONFIG["client_batch_size"],
            shuffle=False,
            worker_init_fn=worker_init_fn,
            drop_last=False,
        )

        max_label = max(phase_label_mapping.values()) if phase_label_mapping else -1
        print(f"Phase {phase} test loader: labels 0-{max_label}")

    print("\n=== PHASE TEST LOADERS READY ===")
    return phase_global_test_loaders
