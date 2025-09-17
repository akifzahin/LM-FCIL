class ClientManager:
    """Manager class to coordinate multiple clients for phase-wise training"""

    def __init__(self, clients, max_phases):
        self.clients = clients
        self.max_phases = max_phases
        self.current_phase = 1
        # Track global state to avoid redundant updates
        self.last_global_num_classes = 0
        self.client_last_known_classes = {client.client_id: 0 for client in clients}

        # CRITICAL FIX: Track globally discovered writers across all clients
        self.globally_discovered_writers = set()

        # NEW: Track global class registry for label remapping
        self.global_class_registry = {}


    def is_client_participating_in_phase(self, client, phase):
        """Check if a client should participate in the given phase"""
        return phase >= client.start_phase and phase in client.phase_loaders

    def update_globally_discovered_writers(self, new_writers):
        """Update the global registry of discovered writers"""
        if new_writers:
            self.globally_discovered_writers.update(new_writers)
            print(f"🌍 Global writers updated: {sorted(self.globally_discovered_writers)}")

            # Propagate to all clients
            for client in self.clients:
                client.update_globally_discovered_writers(self.globally_discovered_writers)
                print(f"→ Passing registry to Client {client.client_id}: {self.global_class_registry}")


    def update_global_class_registry(self, class_registry):
        """Update the global class registry from server"""
        self.global_class_registry = class_registry.copy() if class_registry else {}
        print(f"🗂️  Global class registry updated: {self.global_class_registry}")

    def get_client_updates_for_phase(self, phase, global_weights, global_classes, epochs_per_phase, allow_discovery=False, global_num_classes=None, server_class_registry=None):
        """Get updates from all clients for a specific phase with proper global coordination"""
        client_updates = []

        # Update our class registry from server
        if server_class_registry:
            self.update_global_class_registry(server_class_registry)

        # Determine the current global number of classes
        if global_num_classes is None:
            # Extract from global_weights if available
            if global_weights and 'backbone.classifier.out_features' in global_weights:
                global_num_classes = global_classes
            else:
                global_num_classes = self.last_global_num_classes

        # Update our tracking
        self.last_global_num_classes = global_num_classes

        for client in self.clients:
            if not self.is_client_participating_in_phase(client, phase):
              print(f"⏭️  Client {client.client_id}: Skipping phase {phase} (not participating)")
              continue

            # CRITICAL FIX: Only update model size if global classes increased
            current_client_classes = client._get_model_num_classes()

            if global_num_classes > current_client_classes:
                print(f"Client {client.client_id}: Expanding model from {current_client_classes} to {global_num_classes} classes")
                client.model.expand_classifier(global_num_classes)
                # Keep model on device after expansion
                client.model = client.model.to(client.device)
                client.optimizer = torch.optim.Adam(client.model.parameters(), lr=client.config['local_lr'])

                # Update tracking
                self.client_last_known_classes[client.client_id] = global_num_classes
            if phase >= client.start_phase:
              # FIXED: Pass both global_num_classes and global_class_registry
              update = client.get_client_update_for_phase(
                  phase=phase,
                  global_weights=global_weights,
                  epochs_per_phase=epochs_per_phase,
                  global_num_classes=global_num_classes,
                  global_class_registry=self.global_class_registry  # NEW: Pass class registry
              )

              if update is not None:
                  client_updates.append(update)
                  state_dict = update["weights"]
                  if not os.path.exists(STATE_PATH):
                    os.mkdir(STATE_PATH)
                  state_path = os.path.join(STATE_PATH,f"client_{client.client_id}.pt")

                  # torch.save(state_dict, state_path)
                  print(f"saved to {state_path}")
                  print(f"✅ Client {client.client_id}: Phase {phase} update ready "
                        f"({update['num_samples']} samples, {len(update['discovered_classes'])} writers)")

                  # CRITICAL FIX: Update global writer registry with newly discovered writers
                  if update['discovered_classes']:
                      print(f"🆕 New writers discovered by Client {client.client_id}: {update['discovered_classes']}")
                      self.update_globally_discovered_writers(update['discovered_classes'])
            else:
              continue

        return client_updates

    def update_clients_with_mappings(self, class_mappings, phase=1):
        """Update all clients with new class mappings from server"""
        if not class_mappings:
            return

        # Find the maximum class index across all mappings to determine global model size
        max_global_class = 0
        for client_id, mappings in class_mappings.items():
            if mappings:
                max_global_class = max(max_global_class, max(mappings.values()))

        required_global_classes = max_global_class + 1

        # Update each client's model size to match global requirements
        for client in self.clients:


              current_size = client._get_model_num_classes()

              if required_global_classes > current_size:
                  print(f"Client {client.client_id}: Expanding model from {current_size} to {required_global_classes} classes")
                  client.model.expand_classifier(required_global_classes)

                  # CRITICAL: Keep model on device after expansion
                  client.model = client.model.to(client.device)
                  client.optimizer = torch.optim.Adam(client.model.parameters(), lr=client.config['local_lr'])

                  # Update tracking
                  self.client_last_known_classes[client.client_id] = required_global_classes

    def get_global_model_size(self):
        """Get the current global model size being tracked"""
        return self.last_global_num_classes

    def broadcast_global_weights_to_clients(self, global_weights):
        """Send aggregated global weights to all clients"""
        if not global_weights:
            print("⚠️  No global weights to broadcast")
            return

        print(f"📡 Broadcasting global weights to {len(self.clients)} clients")

        for client in self.clients:
            try:
                # Load the global weights into client's model
                client.model.load_state_dict(global_weights)
                client.model = client.model.to(client.device)

                # Recreate optimizer with updated model parameters
                client.optimizer = torch.optim.Adam(
                    client.model.parameters(),
                    lr=client.config['local_lr']
                )

                print(f"✅ Client {client.client_id}: Global weights loaded successfully")

            except Exception as e:
                print(f"❌ Client {client.client_id}: Failed to load global weights - {e}")
                # You might want to handle this more gracefully
                raise e

    def sync_all_clients_to_global_size(self, global_num_classes):
        """Ensure all clients have models matching the global size"""
        self.last_global_num_classes = global_num_classes

        for client in self.clients:
            current_size = client._get_model_num_classes()
            if current_size != global_num_classes:
                print(f"Client {client.client_id}: Syncing model size from {current_size} to {global_num_classes} classes")
                client.model.expand_classifier(global_num_classes)
                client.model = client.model.to(client.device)
                client.optimizer = torch.optim.Adam(client.model.parameters(), lr=client.config['local_lr'])

                self.client_last_known_classes[client.client_id] = global_num_classes
    def add_client_for_phase(self, new_client, phase, global_weights=None):
        """
        Add a new client to the federated learning system during runtime.

        Args:
            new_client: Fully instantiated WriterIdentificationClient
            phase: Phase number when this client becomes active
            global_weights: Current global model weights to initialize the client
        """
        print(f"🆕 Adding new client '{new_client.client_id}' for phase {phase}")

        # Add to clients list
        self.clients.append(new_client)

        # If global weights provided, sync the new client to current global model
        if global_weights is not None:
            # Get current global model size

           global_num_classes = 0
        if 'backbone.classifier.1.weight' in global_weights:
            global_num_classes = global_weights['backbone.classifier.1.weight'].shape[0]
        elif 'classifier.1.weight' in global_weights:
            global_num_classes = global_weights['classifier.1.weight'].shape[0]

            print(f"  Syncing new client to global model with {global_num_classes} classes")

            # Expand client model to match global size
            if global_num_classes > 0:
                new_client.model.expand_classifier(global_num_classes)
                new_client.model = new_client.model.to(new_client.device)

                # Load global weights
                new_client.model.load_state_dict(global_weights)

                # Recreate optimizer with new parameters
                new_client.optimizer = torch.optim.Adam(
                    new_client.model.parameters(),
                    lr=new_client.config['local_lr']
                )

                print(f"  ✅ Client '{new_client.client_id}' synced with global model")

            print(f"  Total clients now: {len(self.clients)}")
            return True
    def debug_client_states(self):
        """Debug function to check all client model sizes"""
        print(f"\n🔍 ClientManager Debug - Global classes: {self.last_global_num_classes}")
        print(f"🌍 Globally discovered writers: {sorted(self.globally_discovered_writers)}")
        print(f"🗂️  Global class registry: {self.global_class_registry}")
        for client in self.clients:
            current_size = client._get_model_num_classes()  # Changed this line
            last_known = self.client_last_known_classes[client.client_id]
            print(f"  Client {client.client_id}: Model={current_size}, LastKnown={last_known}, Device={next(client.model.parameters()).device}")

