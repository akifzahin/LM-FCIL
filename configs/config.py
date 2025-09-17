import torch

# Federated Learning Configuration
FL_CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "num_clients": 4,
    "total_clients": 4,
    "client_batch_size": 64,
    "local_lr": 0.0005,
    "momentum": 0.9,
    "log_interval": 100,
    "total_rounds": 60,
    "phases": 3,
    "initial_num_classes": 0,
    "epochs_per_phase": 20,
    "num_global_rounds_per_phase": {
        "1": 20,
        "2": 20,
        "3": 20
    }
}

# Training / Dataset Configuration
CONFIG = {
    "batch_size": 512,          # Increase/decrease based on GPU memory
    "learning_rate": 0.001,
    "num_epochs": 1,
    "num_classes": 1989,        # Change dynamically if possible
    "img_size": 64,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "train_split": 0.8,
    "val_split": 0.2
}

# Writers per phase (document meaning)
# Format: { phase_number: [train_clients, val_clients, test_clients] }
writers_per_phase = {
    0: [4, 1, 1],
    1: [4, 1, 1],
    2: [4, 1, 1],
    3: [4, 1, 1]
}
# Note: Phase 0 is for initial setup, phases 1-3 are for incremental learning