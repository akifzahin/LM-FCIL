class FederatedServerEfficient:
    def __init__(self, model_template, state_dict=None, initial_num_classes=0):
        self.global_model = model_template(num_classes=initial_num_classes)
        if state_dict is not None:
            self.global_model.load_state_dict(state_dict)  # FIXED: Use state_dict, not model_template.state_dict()
        self.global_model = self.global_model.to(CONFIG['device'])

        # Dynamic class discovery tracking
        self.global_num_classes = initial_num_classes
        self.class_registry = {}  # Maps writer_id -> global_class_index
        self.next_global_class_id = initial_num_classes  # Next available global class index
        self.current_phase = 1
        # Track training history
        self.round_losses = []
        self.round_accuracies = []
        self.global_accuracies = []

        print(f"✅ Federated Server initialized with {initial_num_classes} classes")

    def safe_load_state_dict(self, model, state_dict, strict=False):
        """
        Safely load state dict, handling size mismatches in classifier layers
        """
        try:
            model.load_state_dict(state_dict, strict=strict)
            return True
        except RuntimeError as e:
            if "size mismatch" in str(e) and ("classifier" in str(e) or "backbone.classifier" in str(e)):
                print(f"⚠️ Size mismatch detected, loading with strict=False: {e}")

                # Load only compatible layers
                model_state = model.state_dict()
                compatible_state = {}

                for key, param in state_dict.items():
                    if key in model_state:
                        if model_state[key].shape == param.shape:
                            compatible_state[key] = param
                        else:
                            print(f"  Skipping {key}: shape mismatch {model_state[key].shape} vs {param.shape}")
                    else:
                        print(f"  Skipping {key}: not found in model")

                # Load compatible layers only
                model.load_state_dict(compatible_state, strict=False)
                print(f"✅ Loaded {len(compatible_state)}/{len(state_dict)} compatible layers")
                return True
            else:
                print(f"❌ Failed to load state dict: {e}")
                return False

    def process_client_updates(self, client_updates):
        """
        Process client updates that may contain new class discoveries

        client_updates: List of dicts containing:
        - 'weights': model state dict
        - 'num_samples': number of training samples
        - 'discovered_classes': list of new writer IDs discovered (if any)
        - 'client_id': identifier for the client
        """
        # Step 1: Process new class discoveries
        global_mapping_changed = False
        client_mappings = {}  # client_id -> {local_class_idx: global_class_idx}

        for update in client_updates:
            client_id = update['client_id']
            discovered_classes = update.get('discovered_classes', [])
            client_mappings[client_id] = {}

            # Register any new classes discovered by this client
            for writer_id in discovered_classes:
                if writer_id not in self.class_registry:
                    # Assign new global class index
                    self.class_registry[writer_id] = self.next_global_class_id
                    self.next_global_class_id += 1
                    global_mapping_changed = True
                    print(f"📝 Registered new writer {writer_id} as global class {self.class_registry[writer_id]}")

        # Step 2: Expand global model if new classes were discovered
        if global_mapping_changed:
            old_num_classes = self.global_num_classes
            self.global_num_classes = self.next_global_class_id

            if self.global_num_classes > old_num_classes:
                print(f"🔄 Expanding global model from {old_num_classes} to {self.global_num_classes} classes")
                self.global_model.expand_classifier(self.global_num_classes)

        # Step 3: Create mapping information for each client
        for update in client_updates:
            client_id = update['client_id']
            # For this implementation, we'll assume clients track their own local->global mapping
            # The server provides the global class registry
            client_mappings[client_id] = dict(self.class_registry)

        return client_mappings, global_mapping_changed

    def get_num_classes(self):
        """Get current number of classes in the global model - EfficientNet compatible"""
        if hasattr(self.global_model, 'backbone') and hasattr(self.global_model.backbone, 'classifier'):
            # EfficientNet structure: model.backbone.classifier
            return self.global_model.current_classes
        elif hasattr(self.global_model, 'classifier'):
            # Original structure: model.classifier
            return self.global_model.current_classes
        return 0

    def federated_averaging_with_dynamic_classes(self, client_updates):
        """
        Enhanced FedAvg that handles dynamic classifier sizes
        """
        # Process new classes first
        client_mappings, mapping_changed = self.process_client_updates(client_updates)

        # Extract just the weights for aggregation (RESTORED ORIGINAL LOGIC)
        client_weights = [update['weights'] for update in client_updates]
        client_samples = [update['num_samples'] for update in client_updates]

        # Perform aggregation with weights only
        aggregated_weights = self.aggregate_weights_dynamic(client_weights, client_samples, client_updates)

        return aggregated_weights, client_mappings, mapping_changed

    def aggregate_weights_dynamic(self, client_weights, client_samples, client_updates):
        """
        Aggregate client weights, handling different classifier sizes
        MINIMAL CHANGE: Added client_updates parameter to access all_phase_classes
        """
        total_samples = sum(client_samples)
        aggregated_weights = {}

        # Get the structure from global model (which has the most up-to-date size)
        global_state = self.global_model.state_dict()

        for key in global_state.keys():
            # Handle both EfficientNet and original classifier structures
            if key in ['classifier.weight', 'backbone.classifier.weight']:
                # Handle classifier weights specially - no averaging for new classes
                aggregated_weights[key] = self.aggregate_classifier_weights(
                    key, client_weights, client_samples, total_samples, client_updates
                )
            elif key in ['classifier.bias', 'backbone.classifier.bias']:
                # Handle classifier bias specially
                aggregated_weights[key] = self.aggregate_classifier_bias(
                    key, client_weights, client_samples, total_samples, client_updates
                )
            else:
                # Regular FedAvg for feature extractor layers
                aggregated_weights[key] = self.aggregate_regular_weights(
                    key, client_weights, client_samples, total_samples
                )

        return aggregated_weights

    def aggregate_classifier_weights(self, key, client_weights, client_samples, total_samples, client_updates):
        """
        Aggregate classifier weights handling different sizes while preserving learned knowledge.
        Uses contribution-weighted averaging to avoid catastrophic forgetting.
        MINIMAL CHANGE: Added client_updates parameter to access all_phase_classes
        """
        global_state = self.global_model.state_dict()
        Cg, D = global_state[key].shape  # Global classifier dimensions
        registry = self.class_registry   # writer_id -> global_idx

        # Start with existing global weights (preserve learned knowledge)
        aggregated = global_state[key].clone().float()

        # Track contributions per class to handle proper weighted averaging
        class_contributions = torch.zeros(Cg, device=aggregated.device)
        class_updates = torch.zeros((Cg, D), device=aggregated.device)

        # Process each client update
        for i, client_w in enumerate(client_weights):
            if key not in client_w:
                continue

            local_w = client_w[key]                           # [C_local, D] - client's classifier weights
            writers = client_updates[i]['all_phase_classes'] # List of writer_ids this client knows
            weight_factor = client_samples[i] / total_samples # Client's contribution weight

            # Ensure we have the writer list
            if len(writers) != local_w.size(0):
                print(f"Warning: Client {client_updates[i]['client_id']} weight size {local_w.size(0)} doesn't match writer count {len(writers)}")
                continue

            # Accumulate updates for each class this client has learned
            for local_idx, writer_id in enumerate(writers):
                if writer_id not in registry:
                    print(f"Warning: Writer {writer_id} not in global registry for client {client_updates[i]['client_id']}")
                    continue

                global_idx = registry[writer_id]

                # Accumulate weighted updates for this class
                class_updates[global_idx] += local_w[local_idx] * weight_factor
                class_contributions[global_idx] += weight_factor

        # Apply accumulated updates only to classes that received contributions
        for global_idx in range(Cg):
            if class_contributions[global_idx] > 0:
                # Use weighted average of contributions for this class
                aggregated[global_idx] = class_updates[global_idx] / class_contributions[global_idx]
            # Classes with no contributions keep their existing weights (no catastrophic forgetting)

        return aggregated.to(dtype=global_state[key].dtype)

    def aggregate_classifier_bias(self, key, client_weights, client_samples, total_samples, client_updates):
        """
        Aggregate classifier bias handling different sizes while preserving learned knowledge.
        Uses contribution-weighted averaging to avoid catastrophic forgetting.
        FIXED: Handle missing 'all_phase_classes' key
        """
        global_state = self.global_model.state_dict()
        Cg = global_state[key].shape[0]  # Global classifier bias dimension
        registry = self.class_registry   # writer_id -> global_idx

        # Start with existing global bias (preserve learned knowledge)
        aggregated_bias = global_state[key].clone().float()

        # Track contributions per class
        class_contributions = torch.zeros(Cg, device=aggregated_bias.device)
        class_updates = torch.zeros(Cg, device=aggregated_bias.device)

        # Process each client update
        for i, client_w in enumerate(client_weights):
            if key not in client_w:
                continue

            local_b = client_w[key]  # [C_local] - client's classifier bias
            weight_factor = client_samples[i] / total_samples  # Client's contribution weight

            # FIXED: Try to get writer list from different possible keys
            writers = None
            if 'all_phase_classes' in client_updates[i]:
                writers = client_updates[i]['all_phase_classes']
            elif 'discovered_classes' in client_updates[i]:
                writers = client_updates[i]['discovered_classes']
            elif 'writers' in client_updates[i]:
                writers = client_updates[i]['writers']
            else:
                # Fallback: skip this client if we can't determine the writer mapping
                print(f"Warning: No writer list found for client {client_updates[i]['client_id']}")
                print(f"  Available keys in client update: {list(client_updates[i].keys())}")
                print(f"  Skipping client {client_updates[i]['client_id']} - cannot determine writer mapping")
                continue

            # Ensure we have the writer list
            if writers is None or len(writers) != local_b.size(0):
                print(f"Warning: Client {client_updates[i]['client_id']} bias size {local_b.size(0)} doesn't match writer count {len(writers) if writers else 0}")
                continue

            # Accumulate updates for each class this client has learned
            for local_idx, writer_id in enumerate(writers):
                if writer_id not in registry:
                    print(f"Warning: Writer {writer_id} not in global registry for client {client_updates[i]['client_id']}")
                    continue

                global_idx = registry[writer_id]

                # Accumulate weighted updates for this class
                class_updates[global_idx] += local_b[local_idx] * weight_factor
                class_contributions[global_idx] += weight_factor
        # Add this after calculating class_contributions
        print(f"Class contributions: {class_contributions.cpu().numpy()}")
        print(f"Classes with zero contributions: {torch.where(class_contributions == 0)[0].cpu().numpy()}")
        # Apply accumulated updates only to classes that received contributions
        for global_idx in range(Cg):
            if class_contributions[global_idx] > 0:
                # Use weighted average of contributions for this class
                aggregated_bias[global_idx] = class_updates[global_idx] / class_contributions[global_idx]
            # Classes with no contributions keep their existing bias (no catastrophic forgetting)

        return aggregated_bias.to(dtype=global_state[key].dtype)


    def aggregate_regular_weights(self, key, client_weights, client_samples, total_samples):
        if True:
            """
            Regular FedAvg for non-classifier layers
            RESTORED ORIGINAL LOGIC
            """
            # Get reference tensor from first client that has this key
            reference_tensor = None
            for client_w in client_weights:
                if key in client_w:
                    reference_tensor = client_w[key]
                    break

            if reference_tensor is None:
                # Fallback to global model's tensor
                return self.global_model.state_dict()[key].clone()

            # Initialize aggregation
            aggregated_tensor = torch.zeros_like(reference_tensor, dtype=torch.float32)

            # Weighted average
            for i, client_w in enumerate(client_weights):
                if key in client_w:
                    weight_factor = client_samples[i] / total_samples
                    client_tensor = client_w[key].to(dtype=torch.float32)
                    aggregated_tensor += client_tensor * weight_factor

            # Convert back to original dtype if needed
            original_dtype = reference_tensor.dtype
            if original_dtype != torch.float32:
                if original_dtype in [torch.int64, torch.long, torch.int32, torch.int]:
                    aggregated_tensor = aggregated_tensor.round().to(dtype=original_dtype)
                else:
                    aggregated_tensor = aggregated_tensor.to(dtype=original_dtype)

            return aggregated_tensor.to(device=reference_tensor.device)

        return self.global_model.state_dict()[key].clone()
    def get_global_weights(self):
        """Return current global model weights"""
        return self.global_model.state_dict()

    def get_global_weights_safe(self, target_num_classes=None):
        """
        Return global model weights, optionally expanded to target_num_classes
        This helps ensure size compatibility when sending to clients
        """
        global_weights = self.global_model.state_dict()

        if target_num_classes is not None:
            current_num_classes = self.get_num_classes()

            if target_num_classes > current_num_classes:
                print(f"🔄 Expanding global weights from {current_num_classes} to {target_num_classes} classes for client compatibility")

                # Handle both EfficientNet and original classifier structures
                if hasattr(self.global_model, 'backbone') and hasattr(self.global_model.backbone, 'classifier'):
                    # EfficientNet structure
                    classifier_weight_key = 'backbone.classifier.weight'
                    classifier_bias_key = 'backbone.classifier.bias'
                else:
                    # Original structure
                    classifier_weight_key = 'classifier.weight'
                    classifier_bias_key = 'classifier.bias'

                if classifier_weight_key in global_weights:
                    old_weight = global_weights[classifier_weight_key]
                    old_bias = global_weights[classifier_bias_key]

                    # Create expanded tensors
                    new_weight = torch.zeros((target_num_classes, old_weight.shape[1]),
                                           dtype=old_weight.dtype, device=old_weight.device)
                    new_bias = torch.zeros(target_num_classes,
                                         dtype=old_bias.dtype, device=old_bias.device)

                    # Copy existing weights
                    if current_num_classes > 0:
                        new_weight[:current_num_classes, :] = old_weight
                        new_bias[:current_num_classes] = old_bias

                    # Initialize new classes with small random weights
                    if target_num_classes > current_num_classes:
                        torch.nn.init.normal_(new_weight[current_num_classes:, :], mean=0.0, std=0.01)
                        torch.nn.init.constant_(new_bias[current_num_classes:], 0.0)

                    # Update the weights dict
                    global_weights = global_weights.copy()  # Don't modify original
                    global_weights[classifier_weight_key] = new_weight
                    global_weights[classifier_bias_key] = new_bias

        return global_weights

    def get_class_mapping(self):
        """Return the current global class mapping"""
        return dict(self.class_registry)

    def get_global_model_info(self):
        """Return information about the global model"""
        return {
            'num_classes': self.global_num_classes,
            'class_registry': dict(self.class_registry),
            'next_class_id': self.next_global_class_id
        }
    def update_global_model(self, aggregated_weights):
        """Update global model with aggregated weights - handles size mismatches"""

        # Get current global model state
        current_state = self.global_model.state_dict()

        # Check for size mismatches in classifier layers - handle EfficientNet structure
        classifier_weight_key = 'backbone.classifier.weight'
        classifier_bias_key = 'backbone.classifier.bias'

        if classifier_weight_key in aggregated_weights and classifier_weight_key in current_state:
            current_classifier_shape = current_state[classifier_weight_key].shape
            new_classifier_shape = aggregated_weights[classifier_weight_key].shape

            print(f"🔍 Global model update check:")
            print(f"  Current global model classifier: {current_classifier_shape}")
            print(f"  Aggregated weights classifier: {new_classifier_shape}")

            if current_classifier_shape != new_classifier_shape:
                print(f"⚠️ Size mismatch detected!")

                # Option 1: Expand aggregated weights to match global model size
                if new_classifier_shape[0] < current_classifier_shape[0]:
                    print(f"  Expanding aggregated weights from {new_classifier_shape[0]} to {current_classifier_shape[0]} classes")

                    # Get device and dtype from current model
                    device = current_state[classifier_weight_key].device
                    dtype = current_state[classifier_weight_key].dtype

                    # Create expanded weights by padding with zeros
                    expanded_classifier_weight = torch.zeros(current_classifier_shape, dtype=dtype, device=device)
                    expanded_classifier_bias = torch.zeros(current_classifier_shape[0], dtype=dtype, device=device)

                    # Copy existing weights (move to correct device first)
                    expanded_classifier_weight[:new_classifier_shape[0], :] = aggregated_weights[classifier_weight_key].to(device)
                    expanded_classifier_bias[:new_classifier_shape[0]] = aggregated_weights[classifier_bias_key].to(device)

                    # Initialize new classes with small random weights
                    if new_classifier_shape[0] < current_classifier_shape[0]:
                        torch.nn.init.normal_(expanded_classifier_weight[new_classifier_shape[0]:, :], mean=0.0, std=0.01)
                        torch.nn.init.constant_(expanded_classifier_bias[new_classifier_shape[0]:], 0.0)

                    # Update aggregated weights
                    aggregated_weights[classifier_weight_key] = expanded_classifier_weight
                    aggregated_weights[classifier_bias_key] = expanded_classifier_bias

                    print(f"  ✅ Expanded aggregated weights to match global model")

                # Option 2: Expand global model to match aggregated weights
                elif new_classifier_shape[0] > current_classifier_shape[0]:
                    print(f"  Expanding global model from {current_classifier_shape[0]} to {new_classifier_shape[0]} classes")
                    self.global_model.expand_classifier(new_classifier_shape[0])
                    print(f"  ✅ Expanded global model to match aggregated weights")

        # Now load the state dict using safe loading
        success = self.safe_load_state_dict(self.global_model, aggregated_weights)

        # Debug: Compare weight magnitudes between old and new classes
        classifier_weights = self.global_model.state_dict()[classifier_weight_key]
        print(f"\n=== WEIGHT MAGNITUDE COMPARISON ===")
        old_class_norms = [torch.norm(classifier_weights[i]).item() for i in range(min(12, classifier_weights.shape[0]))]
        new_class_norms = [torch.norm(classifier_weights[i]).item() for i in range(12, min(24, classifier_weights.shape[0]))]

        print(f"Old classes (0-11) weight norms: {[f'{x:.4f}' for x in old_class_norms]}")
        print(f"New classes (12-23) weight norms: {[f'{x:.4f}' for x in new_class_norms]}")
        print(f"Average old class norm: {sum(old_class_norms)/len(old_class_norms) if old_class_norms else 0:.4f}")
        print(f"Average new class norm: {sum(new_class_norms)/len(new_class_norms) if new_class_norms else 0:.4f}")

        if success:
            print(f"✅ Global model updated successfully")
            # After successful model update, normalize classifier weights
            classifier_weights = self.global_model.state_dict()[classifier_weight_key]
            classifier_bias = self.global_model.state_dict()[classifier_bias_key]

            # Normalize each class's weight vector to have similar magnitude
            normalized_weights = torch.nn.functional.normalize(classifier_weights, dim=1) * 0.1  # Scale down
            normalized_bias = torch.zeros_like(classifier_bias)  # Reset bias to zero

            # Update the model
            state_dict = self.global_model.state_dict()
            state_dict[classifier_weight_key] = normalized_weights
            state_dict[classifier_bias_key] = normalized_bias
            self.global_model.load_state_dict(state_dict)
        else:
            print(f"⚠️ Global model update had issues")

    def global_evaluate(self, test_loader):
        """
        Evaluate the global_model only on test samples whose raw writer-ID is registered.
        Remap each raw writer-ID → the global index before computing loss/accuracy.
        """
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, raw_labels in test_loader:
                # images: tensor on CPU, raw_labels: tensor on CPU
                batch_size = raw_labels.size(0)
                Cg = self.global_model.backbone.classifier.out_features

                # Build mask on CPU: which raw_labels are in class_registry?
                mask_cpu = torch.zeros(batch_size, dtype=torch.bool)
                mapped_cpu = torch.zeros(batch_size, dtype=torch.long)

                for i, raw in enumerate(raw_labels.tolist()):
                    if raw in self.class_registry:
                        mask_cpu[i] = True
                        mapped_cpu[i] = self.class_registry[raw]

                if mask_cpu.sum().item() == 0:
                    # No registered writers in this batch → skip
                    continue

                # Keep only the masked samples (still on CPU)
                imgs_cpu   = images[mask_cpu]                 # still on CPU
                targets_cpu = mapped_cpu[mask_cpu]             # still on CPU

                # Now move sliced tensors to GPU (or whatever CONFIG['device'] is)
                imgs   = imgs_cpu.to(CONFIG['device']).float()
                targets = targets_cpu.to(CONFIG['device']).long()

                # Forward + loss on device
                outputs = self.global_model(imgs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                total_loss += loss.item() * imgs.size(0)

                _, preds = torch.max(outputs, 1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                print(f"Prediction distribution: {torch.bincount(preds.cpu(), minlength=Cg)}")
                print(f"Target distribution:     {torch.bincount(targets.cpu(), minlength=Cg)}")

        if total > 0:
            avg_loss = total_loss / total
            accuracy = 100.0 * correct / total
        else:
            avg_loss, accuracy = float('inf'), 0.0

        return avg_loss, accuracy

class FederatedServerMobile:
    def __init__(self, model_template, state_dict=None, initial_num_classes=0):
        self.global_model = model_template(num_classes=initial_num_classes)
        if state_dict is not None:
            self.global_model.load_state_dict(state_dict)  # FIXED: Use state_dict, not model_template.state_dict()
        self.global_model = self.global_model.to(CONFIG['device'])

        # Dynamic class discovery tracking
        self.global_num_classes = initial_num_classes
        self.class_registry = {}  # Maps writer_id -> global_class_index
        self.next_global_class_id = initial_num_classes  # Next available global class index

        # Track training history
        self.round_losses = []
        self.round_accuracies = []
        self.global_accuracies = []

        print(f"✅ Federated Server initialized with {initial_num_classes} classes")

    def get_classifier_keys(self):
        """Get the correct classifier keys for the model architecture"""
        # For MobileNet: backbone.classifier.3.weight and backbone.classifier.3.bias
        # For EfficientNet: backbone.classifier.weight and backbone.classifier.bias
        state_dict = self.global_model.state_dict()

        # Try MobileNet structure first
        if 'backbone.classifier.3.weight' in state_dict:
            return 'backbone.classifier.3.weight', 'backbone.classifier.3.bias'
        # Fallback to EfficientNet structure
        elif 'backbone.classifier.weight' in state_dict:
            return 'backbone.classifier.weight', 'backbone.classifier.bias'
        else:
            # Find any classifier layers
            classifier_keys = [k for k in state_dict.keys() if 'classifier' in k and 'weight' in k]
            if classifier_keys:
                weight_key = classifier_keys[0]
                bias_key = weight_key.replace('weight', 'bias')
                return weight_key, bias_key
            else:
                raise ValueError("Could not find classifier layers in model")

    def safe_load_state_dict(self, model, state_dict, strict=False):
        """
        Safely load state dict, handling size mismatches in classifier layers
        """
        try:
            model.load_state_dict(state_dict, strict=strict)
            return True
        except RuntimeError as e:
            if "size mismatch" in str(e) and "classifier" in str(e):
                print(f"⚠️ Size mismatch detected, loading with strict=False: {e}")

                # Load only compatible layers
                model_state = model.state_dict()
                compatible_state = {}

                for key, param in state_dict.items():
                    if key in model_state:
                        if model_state[key].shape == param.shape:
                            compatible_state[key] = param
                        else:
                            print(f"  Skipping {key}: shape mismatch {model_state[key].shape} vs {param.shape}")
                    else:
                        print(f"  Skipping {key}: not found in model")

                # Load compatible layers only
                model.load_state_dict(compatible_state, strict=False)
                print(f"✅ Loaded {len(compatible_state)}/{len(state_dict)} compatible layers")
                return True
            else:
                print(f"❌ Failed to load state dict: {e}")
                return False

    def process_client_updates(self, client_updates):
        """
        Process client updates that may contain new class discoveries

        client_updates: List of dicts containing:
        - 'weights': model state dict
        - 'num_samples': number of training samples
        - 'discovered_classes': list of new writer IDs discovered (if any)
        - 'client_id': identifier for the client
        """
        # Step 1: Process new class discoveries
        global_mapping_changed = False
        client_mappings = {}  # client_id -> {local_class_idx: global_class_idx}

        for update in client_updates:
            client_id = update['client_id']
            discovered_classes = update.get('discovered_classes', [])
            client_mappings[client_id] = {}

            # Register any new classes discovered by this client
            for writer_id in discovered_classes:
                if writer_id not in self.class_registry:
                    # Assign new global class index
                    self.class_registry[writer_id] = self.next_global_class_id
                    self.next_global_class_id += 1
                    global_mapping_changed = True
                    print(f"📝 Registered new writer {writer_id} as global class {self.class_registry[writer_id]}")

        # Step 2: Expand global model if new classes were discovered
        if global_mapping_changed:
            old_num_classes = self.global_num_classes
            self.global_num_classes = self.next_global_class_id

            if self.global_num_classes > old_num_classes:
                print(f"🔄 Expanding global model from {old_num_classes} to {self.global_num_classes} classes")
                self.global_model.expand_classifier(self.global_num_classes)

        # Step 3: Create mapping information for each client
        for update in client_updates:
            client_id = update['client_id']
            # For this implementation, we'll assume clients track their own local->global mapping
            # The server provides the global class registry
            client_mappings[client_id] = dict(self.class_registry)

        return client_mappings, global_mapping_changed

    def get_num_classes(self):
        """Get current number of classes in the global model"""
        if hasattr(self.global_model, 'classifier'):
            return self.global_model.classifier.out_features
        return 0

    def federated_averaging_with_dynamic_classes(self, client_updates):
        """
        Enhanced FedAvg that handles dynamic classifier sizes
        """
        # Process new classes first
        client_mappings, mapping_changed = self.process_client_updates(client_updates)

        # Extract just the weights for aggregation (RESTORED ORIGINAL LOGIC)
        client_weights = [update['weights'] for update in client_updates]
        client_samples = [update['num_samples'] for update in client_updates]

        # Perform aggregation with weights only
        aggregated_weights = self.aggregate_weights_dynamic(client_weights, client_samples, client_updates)

        return aggregated_weights, client_mappings, mapping_changed

    def aggregate_weights_dynamic(self, client_weights, client_samples, client_updates):
        """
        Aggregate client weights, handling different classifier sizes
        MINIMAL CHANGE: Added client_updates parameter to access all_phase_classes
        """
        total_samples = sum(client_samples)
        aggregated_weights = {}

        # Get the structure from global model (which has the most up-to-date size)
        global_state = self.global_model.state_dict()

        # Get correct classifier keys for this model architecture
        classifier_weight_key, classifier_bias_key = self.get_classifier_keys()

        for key in global_state.keys():
            if key == classifier_weight_key:
                # Handle classifier weights specially - no averaging for new classes
                aggregated_weights[key] = self.aggregate_classifier_weights(
                    key, client_weights, client_samples, total_samples, client_updates
                )
            elif key == classifier_bias_key:
                # Handle classifier bias specially
                aggregated_weights[key] = self.aggregate_classifier_bias(
                    key, client_weights, client_samples, total_samples, client_updates
                )
            else:
                # Regular FedAvg for feature extractor layers
                aggregated_weights[key] = self.aggregate_regular_weights(
                    key, client_weights, client_samples, total_samples
                )

        return aggregated_weights

    def aggregate_classifier_weights(self, key, client_weights, client_samples, total_samples, client_updates):
        """
        Aggregate classifier weights handling different sizes while preserving learned knowledge.
        Uses contribution-weighted averaging to avoid catastrophic forgetting.
        MINIMAL CHANGE: Added client_updates parameter to access all_phase_classes
        """
        global_state = self.global_model.state_dict()
        Cg, D = global_state[key].shape  # Global classifier dimensions
        registry = self.class_registry   # writer_id -> global_idx

        # Start with existing global weights (preserve learned knowledge)
        aggregated = global_state[key].clone().float()

        # Track contributions per class to handle proper weighted averaging
        class_contributions = torch.zeros(Cg, device=aggregated.device)
        class_updates = torch.zeros((Cg, D), device=aggregated.device)

        # Process each client update
        for i, client_w in enumerate(client_weights):
            if key not in client_w:
                continue

            local_w = client_w[key]                           # [C_local, D] - client's classifier weights
            writers = client_updates[i]['all_phase_classes'] # List of writer_ids this client knows
            weight_factor = client_samples[i] / total_samples # Client's contribution weight

            # Ensure we have the writer list
            if len(writers) != local_w.size(0):
                print(f"Warning: Client {client_updates[i]['client_id']} weight size {local_w.size(0)} doesn't match writer count {len(writers)}")
                continue

            # Accumulate updates for each class this client has learned
            for local_idx, writer_id in enumerate(writers):
                if writer_id not in registry:
                    print(f"Warning: Writer {writer_id} not in global registry for client {client_updates[i]['client_id']}")
                    continue

                global_idx = registry[writer_id]

                # Accumulate weighted updates for this class
                class_updates[global_idx] += local_w[local_idx] * weight_factor
                class_contributions[global_idx] += weight_factor

        # Apply accumulated updates only to classes that received contributions
        for global_idx in range(Cg):
            if class_contributions[global_idx] > 0:
                # Use weighted average of contributions for this class
                aggregated[global_idx] = class_updates[global_idx] / class_contributions[global_idx]
            # Classes with no contributions keep their existing weights (no catastrophic forgetting)

        return aggregated.to(dtype=global_state[key].dtype)

    def aggregate_classifier_bias(self, key, client_weights, client_samples, total_samples, client_updates):
        """
        Aggregate classifier bias handling different sizes while preserving learned knowledge.
        Uses contribution-weighted averaging to avoid catastrophic forgetting.
        FIXED: Handle missing 'all_phase_classes' key
        """
        global_state = self.global_model.state_dict()
        Cg = global_state[key].shape[0]  # Global classifier bias dimension
        registry = self.class_registry   # writer_id -> global_idx

        # Start with existing global bias (preserve learned knowledge)
        aggregated_bias = global_state[key].clone().float()

        # Track contributions per class
        class_contributions = torch.zeros(Cg, device=aggregated_bias.device)
        class_updates = torch.zeros(Cg, device=aggregated_bias.device)

        # Process each client update
        for i, client_w in enumerate(client_weights):
            if key not in client_w:
                continue

            local_b = client_w[key]  # [C_local] - client's classifier bias
            weight_factor = client_samples[i] / total_samples  # Client's contribution weight

            # FIXED: Try to get writer list from different possible keys
            writers = None
            if 'all_phase_classes' in client_updates[i]:
                writers = client_updates[i]['all_phase_classes']
            elif 'discovered_classes' in client_updates[i]:
                writers = client_updates[i]['discovered_classes']
            elif 'writers' in client_updates[i]:
                writers = client_updates[i]['writers']
            else:
                # Fallback: skip this client if we can't determine the writer mapping
                print(f"Warning: No writer list found for client {client_updates[i]['client_id']}")
                print(f"  Available keys in client update: {list(client_updates[i].keys())}")
                print(f"  Skipping client {client_updates[i]['client_id']} - cannot determine writer mapping")
                continue

            # Ensure we have the writer list
            if writers is None or len(writers) != local_b.size(0):
                print(f"Warning: Client {client_updates[i]['client_id']} bias size {local_b.size(0)} doesn't match writer count {len(writers) if writers else 0}")
                continue

            # Accumulate updates for each class this client has learned
            for local_idx, writer_id in enumerate(writers):
                if writer_id not in registry:
                    print(f"Warning: Writer {writer_id} not in global registry for client {client_updates[i]['client_id']}")
                    continue

                global_idx = registry[writer_id]

                # Accumulate weighted updates for this class
                class_updates[global_idx] += local_b[local_idx] * weight_factor
                class_contributions[global_idx] += weight_factor
        # Add this after calculating class_contributions
        print(f"Class contributions: {class_contributions.cpu().numpy()}")
        print(f"Classes with zero contributions: {torch.where(class_contributions == 0)[0].cpu().numpy()}")
        # Apply accumulated updates only to classes that received contributions
        for global_idx in range(Cg):
            if class_contributions[global_idx] > 0:
                # Use weighted average of contributions for this class
                aggregated_bias[global_idx] = class_updates[global_idx] / class_contributions[global_idx]
            # Classes with no contributions keep their existing bias (no catastrophic forgetting)

        return aggregated_bias.to(dtype=global_state[key].dtype)

    def aggregate_regular_weights(self, key, client_weights, client_samples, total_samples):
        """
        Regular FedAvg for non-classifier layers
        RESTORED ORIGINAL LOGIC
        """
        # Get reference tensor from first client that has this key
        reference_tensor = None
        for client_w in client_weights:
            if key in client_w:
                reference_tensor = client_w[key]
                break

        if reference_tensor is None:
            # Fallback to global model's tensor
            return self.global_model.state_dict()[key].clone()

        # Initialize aggregation
        aggregated_tensor = torch.zeros_like(reference_tensor, dtype=torch.float32)

        # Weighted average
        for i, client_w in enumerate(client_weights):
            if key in client_w:
                weight_factor = client_samples[i] / total_samples
                client_tensor = client_w[key].to(dtype=torch.float32)
                aggregated_tensor += client_tensor * weight_factor

        # Convert back to original dtype if needed
        original_dtype = reference_tensor.dtype
        if original_dtype != torch.float32:
            if original_dtype in [torch.int64, torch.long, torch.int32, torch.int]:
                aggregated_tensor = aggregated_tensor.round().to(dtype=original_dtype)
            else:
                aggregated_tensor = aggregated_tensor.to(dtype=original_dtype)

        return aggregated_tensor.to(device=reference_tensor.device)

    def get_global_weights(self):
        """Return current global model weights"""
        return self.global_model.state_dict()

    def get_global_weights_safe(self, target_num_classes=None):
        """
        Return global model weights, optionally expanded to target_num_classes
        This helps ensure size compatibility when sending to clients
        """
        global_weights = self.global_model.state_dict()

        if target_num_classes is not None:
            current_num_classes = self.get_num_classes()

            if target_num_classes > current_num_classes:
                print(f"🔄 Expanding global weights from {current_num_classes} to {target_num_classes} classes for client compatibility")

                # Get correct classifier keys
                classifier_weight_key, classifier_bias_key = self.get_classifier_keys()

                if classifier_weight_key in global_weights:
                    old_weight = global_weights[classifier_weight_key]
                    old_bias = global_weights[classifier_bias_key]

                    # Create expanded tensors
                    new_weight = torch.zeros((target_num_classes, old_weight.shape[1]),
                                           dtype=old_weight.dtype, device=old_weight.device)
                    new_bias = torch.zeros(target_num_classes,
                                         dtype=old_bias.dtype, device=old_bias.device)

                    # Copy existing weights
                    if current_num_classes > 0:
                        new_weight[:current_num_classes, :] = old_weight
                        new_bias[:current_num_classes] = old_bias

                    # Initialize new classes with small random weights
                    if target_num_classes > current_num_classes:
                        torch.nn.init.normal_(new_weight[current_num_classes:, :], mean=0.0, std=0.01)
                        torch.nn.init.constant_(new_bias[current_num_classes:], 0.0)

                    # Update the weights dict
                    global_weights = global_weights.copy()  # Don't modify original
                    global_weights[classifier_weight_key] = new_weight
                    global_weights[classifier_bias_key] = new_bias

        return global_weights

    def get_class_mapping(self):
        """Return the current global class mapping"""
        return dict(self.class_registry)

    def get_global_model_info(self):
        """Return information about the global model"""
        return {
            'num_classes': self.global_num_classes,
            'class_registry': dict(self.class_registry),
            'next_class_id': self.next_global_class_id
        }

    def update_global_model(self, aggregated_weights):
        """Update global model with aggregated weights - handles size mismatches"""

        # Get current global model state
        current_state = self.global_model.state_dict()

        # Get correct classifier keys for this model architecture
        classifier_weight_key, classifier_bias_key = self.get_classifier_keys()

        if classifier_weight_key in aggregated_weights and classifier_weight_key in current_state:
            current_classifier_shape = current_state[classifier_weight_key].shape
            new_classifier_shape = aggregated_weights[classifier_weight_key].shape

            print(f"🔍 Global model update check:")
            print(f"  Current global model classifier: {current_classifier_shape}")
            print(f"  Aggregated weights classifier: {new_classifier_shape}")

            if current_classifier_shape != new_classifier_shape:
                print(f"⚠️ Size mismatch detected!")

                # Option 1: Expand aggregated weights to match global model size
                if new_classifier_shape[0] < current_classifier_shape[0]:
                    print(f"  Expanding aggregated weights from {new_classifier_shape[0]} to {current_classifier_shape[0]} classes")

                    # Get device and dtype from current model
                    device = current_state[classifier_weight_key].device
                    dtype = current_state[classifier_weight_key].dtype

                    # Create expanded weights by padding with zeros
                    expanded_classifier_weight = torch.zeros(current_classifier_shape, dtype=dtype, device=device)
                    expanded_classifier_bias = torch.zeros(current_classifier_shape[0], dtype=dtype, device=device)

                    # Copy existing weights (move to correct device first)
                    expanded_classifier_weight[:new_classifier_shape[0], :] = aggregated_weights[classifier_weight_key].to(device)
                    expanded_classifier_bias[:new_classifier_shape[0]] = aggregated_weights[classifier_bias_key].to(device)

                    # Initialize new classes with small random weights
                    if new_classifier_shape[0] < current_classifier_shape[0]:
                        torch.nn.init.normal_(expanded_classifier_weight[new_classifier_shape[0]:, :], mean=0.0, std=0.01)
                        torch.nn.init.constant_(expanded_classifier_bias[new_classifier_shape[0]:], 0.0)

                    # Update aggregated weights
                    aggregated_weights[classifier_weight_key] = expanded_classifier_weight
                    aggregated_weights[classifier_bias_key] = expanded_classifier_bias

                    print(f"  ✅ Expanded aggregated weights to match global model")

                # Option 2: Expand global model to match aggregated weights
                elif new_classifier_shape[0] > current_classifier_shape[0]:
                    print(f"  Expanding global model from {current_classifier_shape[0]} to {new_classifier_shape[0]} classes")
                    self.global_model.expand_classifier(new_classifier_shape[0])
                    print(f"  ✅ Expanded global model to match aggregated weights")

        # Now load the state dict using safe loading
        success = self.safe_load_state_dict(self.global_model, aggregated_weights)

        # Debug: Compare weight magnitudes between old and new classes
        classifier_weights = self.global_model.state_dict()[classifier_weight_key]
        print(f"\n=== WEIGHT MAGNITUDE COMPARISON ===")
        old_class_norms = [torch.norm(classifier_weights[i]).item() for i in range(min(12, classifier_weights.shape[0]))]
        new_class_norms = [torch.norm(classifier_weights[i]).item() for i in range(12, min(24, classifier_weights.shape[0]))]

        print(f"Old classes (0-11) weight norms: {[f'{x:.4f}' for x in old_class_norms]}")
        print(f"New classes (12-23) weight norms: {[f'{x:.4f}' for x in new_class_norms]}")
        print(f"Average old class norm: {sum(old_class_norms)/len(old_class_norms) if old_class_norms else 0:.4f}")
        print(f"Average new class norm: {sum(new_class_norms)/len(new_class_norms) if new_class_norms else 0:.4f}")

        if success:
            print(f"✅ Global model updated successfully")
        else:
            print(f"⚠️ Global model update had issues")

    def global_evaluate(self, test_loader):
        """
        Evaluate the global_model only on test samples whose raw writer-ID is registered.
        Remap each raw writer-ID → the global index before computing loss/accuracy.
        """
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, raw_labels in test_loader:
                # images: tensor on CPU, raw_labels: tensor on CPU
                batch_size = raw_labels.size(0)
                Cg = self.global_model.classifier.out_features

                # Build mask on CPU: which raw_labels are in class_registry?
                mask_cpu = torch.zeros(batch_size, dtype=torch.bool)
                mapped_cpu = torch.zeros(batch_size, dtype=torch.long)

                for i, raw in enumerate(raw_labels.tolist()):
                    if raw in self.class_registry:
                        mask_cpu[i] = True
                        mapped_cpu[i] = self.class_registry[raw]

                if mask_cpu.sum().item() == 0:
                    # No registered writers in this batch → skip
                    continue

                # Keep only the masked samples (still on CPU)
                imgs_cpu   = images[mask_cpu]                 # still on CPU
                targets_cpu = mapped_cpu[mask_cpu]             # still on CPU

                # Now move sliced tensors to GPU (or whatever CONFIG['device'] is)
                imgs   = imgs_cpu.to(CONFIG['device']).float()
                targets = targets_cpu.to(CONFIG['device']).long()

                # Forward + loss on device
                outputs = self.global_model(imgs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                total_loss += loss.item() * imgs.size(0)

                _, preds = torch.max(outputs, 1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                print(f"Prediction distribution: {torch.bincount(preds.cpu(), minlength=Cg)}")
                print(f"Target distribution:     {torch.bincount(targets.cpu(), minlength=Cg)}")

        if total > 0:
            avg_loss = total_loss / total
            accuracy = 100.0 * correct / total
        else:
            avg_loss, accuracy = float('inf'), 0.0

        return avg_loss, accuracy

