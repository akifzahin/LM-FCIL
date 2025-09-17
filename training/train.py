import torch
from torch.utils.data import DataLoader
import json

def initialize_server(template_model='WriterIdentificationMobileModel'):
    """
    Initialize the federated server with the specified model.
    """
    if template_model == 'WriterIdentificationMobileModel':
        from your_models_module import WriterIdentificationMobileModel, FederatedServerMobile
        server = FederatedServerMobile(WriterIdentificationMobileModel)
    else:
        from your_models_module import WriterIdentificationEfficientModel, FederatedServerEfficient
        server = FederatedServerEfficient(WriterIdentificationEfficientModel)
    
    server.class_registry = {}
    print("Federated server initialized")
    return server

def train_federated(server, clients, client_manager, FL_CONFIG, phase_global_test_loaders, device):
    """
    Main federated training loop.
    Returns the trained server, accuracies, and comprehensive metrics.
    """
    global_round = 0
    best_val_acc = 0
    accuracies = []
    phase_comprehensive_metrics = {}

    for phase in range(1, FL_CONFIG['phases'] + 1):
        print(f"\n{'='*60}")
        print(f"STARTING PHASE {phase}/{FL_CONFIG['phases']}")
        print(f"{'='*60}")
        
        global_test_loader = phase_global_test_loaders[phase]
        rounds_in_phase = FL_CONFIG['num_global_rounds_per_phase']

        for round_in_phase in range(1, rounds_in_phase[str(phase)] + 1):
            global_round += 1
            is_discovery = (round_in_phase == 1)
            print(f"\nGlobal Round {global_round} - Phase {phase} - Round {round_in_phase}")

            # Fetch client updates
            client_updates = client_manager.get_client_updates_for_phase(
                phase=phase,
                global_weights=server.get_global_weights(),
                global_classes=server.get_num_classes(),
                epochs_per_phase=FL_CONFIG['epochs_per_phase'],
                allow_discovery=is_discovery,
                global_num_classes=server.get_num_classes(),
                server_class_registry=None if is_discovery else server.class_registry
            )

            if not client_updates:
                print("No updates received this round, skipping.")
                continue

            # Aggregate client updates and update global model
            aggregated_weights, client_mappings, mapping_changed = server.federated_averaging_with_dynamic_classes(client_updates)
            server.update_global_model(aggregated_weights)

            # Save global model weights and class registry
            torch.save(server.get_global_weights(), f"/content/global_weights_round.pt", _use_new_zipfile_serialization=True)
            with open("/content/class_registry.json", "w") as f:
                json.dump(server.class_registry, f)

            # Broadcast new class mappings if changed
            if mapping_changed and client_mappings:
                client_manager.update_clients_with_mappings(client_mappings, phase)

            # Evaluate global model
            is_last_round = (round_in_phase == rounds_in_phase[str(phase)])
            if is_last_round:
                metrics = calculate_comprehensive_metrics(server, global_test_loader, device)
                phase_comprehensive_metrics[phase] = metrics
                acc = metrics.get('accuracy', 0)
            else:
                _, acc = server.global_evaluate(global_test_loader)

            # Track best accuracy
            if acc > best_val_acc:
                best_val_acc = acc
                save_model_traced(server.global_model)

            accuracies.append((phase, global_round, acc))

        print(f"\n✅ PHASE {phase} COMPLETED - Current global test accuracy: {acc:.2f}%")

    return server, accuracies, phase_comprehensive_metrics

def main(clients, client_manager, FL_CONFIG, phase_global_test_loaders, device):
    """
    Run the full federated training pipeline.
    """
    server = initialize_server()
    trained_server, accuracies, metrics = train_federated(
        server, clients, client_manager, FL_CONFIG, phase_global_test_loaders, device
    )
    print("\n🏁 FEDERATED TRAINING COMPLETED!")
    return trained_server, accuracies, metrics
