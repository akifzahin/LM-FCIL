class FederatedClient2:
    def __init__(self, client_id, dataset, model_template, config, start_phase=1):
        self.client_id = client_id
        self.dataset = dataset  # This is a CustomSubset with label remapping
        self.config = config

        # CRITICAL FIX: Set device first
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Client {client_id}: Using device {self.device}")

        # Phase-wise training setup
        self.max_phases = config.get('phases', 1)
        self.current_phase = 1
        self.phase_datasets = {}
        self.phase_loaders = {}
        self.start_phase = start_phase

        # Training history tracking
        self.training_history = {
            'phases': {},
            'discovered_writers': set(),
            'local_performance': []
        }

        # CRITICAL FIX: Track globally discovered writers to avoid re-discovery
        self.globally_discovered_writers = set()

        # Initialize phase data partitioning first
        self._initialize_phase_datasets()

        # FIXED: Initialize model with SAME size as global model
        # Check for MobileNet structure
        if hasattr(model_template, 'classifier') and hasattr(model_template.classifier, 'out_features'):
            global_num_classes = model_template.classifier.out_features
        else:
            global_num_classes = config.get('initial_num_classes', 1)

        self.model = WriterIdentificationMobileModel(num_classes=global_num_classes)

        # Load template weights if provided
        if hasattr(model_template, 'state_dict'):
            self.model.load_state_dict(model_template.state_dict())

        # CRITICAL FIX: Move model to device
        self.model = self.model.to(self.device)

        # CRITICAL FIX: Move criterion to device and set up optimizer
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['local_lr'])
        self.scheduler = StepLR(self.optimizer, step_size=8, gamma=0.8)

        print(f"Client {client_id}: Initialized with {self.max_phases} phases")
        print(f"  Model initialized with {global_num_classes} classes (matching global model)")
        print(f"  Model device: {next(self.model.parameters()).device}")
        print(f"  Criterion device: {self.device}")

        for phase, dataset in self.phase_datasets.items():
            print(f"  Phase {phase}: {len(dataset)} samples")

    def _get_model_num_classes(model):
        """Helper to get the number of output classes based on model type."""
        # SwiftFormer model structure: classifier_head is Sequential with Linear at index 3
        if hasattr(model, 'classifier_head') and isinstance(model.classifier_head, nn.Sequential):
            # For SwiftFormer models with classifier_head
            final_layer = model.classifier_head[-1]  # Last layer (Linear)
            return getattr(final_layer, 'out_features', 0)
        elif hasattr(model, 'backbone') and hasattr(model.backbone, 'classifier'):
            # Original MobileNet/SqueezeNet structure: backbone.classifier is Sequential
            if isinstance(model.backbone.classifier, nn.Sequential):
                layer = model.backbone.classifier[-1]  # Last layer
                return getattr(layer, 'out_channels', getattr(layer, 'out_features', 0))
            else:
                return model.backbone.classifier.out_features
        elif hasattr(model, 'classifier'):
            # Direct classifier structure
            if isinstance(model.classifier, nn.Sequential):
                final_layer = model.classifier[-1]  # Last layer
                return getattr(final_layer, 'out_channels', getattr(final_layer, 'out_features', 0))
            else:
                return model.classifier.out_features
        else:
            return 0

    def _initialize_phase_datasets(self):
        """FIXED: Partition client dataset into phases without double label mapping"""
        # Get unique ORIGINAL writer IDs from the client dataset
        original_writers = self.dataset.get_original_labels()
        original_writers.sort()
        total_writers = len(original_writers)

        print(f"Client {self.client_id}: Original writers = {original_writers}")

        # CHANGE: Get writers per phase from config or calculate default
        if 'writers_per_phase' in self.config:
            writers_per_phase_list = self.config['writers_per_phase']
        else:
            writers_per_phase = max(1, total_writers // (self.max_phases - self.start_phase + 1))
            writers_per_phase_list = [writers_per_phase] * (self.max_phases - self.start_phase + 1)

        phase_writer_mapping = {}

        # CHANGE: Only create phases from start_phase onwards with controlled writer introduction (CUMULATIVE)
        for i, phase in enumerate(range(self.start_phase, self.max_phases + 1)):
            if i < len(writers_per_phase_list):
                # Cumulative: sum up all writers up to this phase
                cumulative_writers = sum(writers_per_phase_list[:i+1])
            else:
                cumulative_writers = total_writers  # All remaining writers

            cumulative_writers = min(cumulative_writers, total_writers)  # Don't exceed total
            phase_writer_mapping[phase] = original_writers[:cumulative_writers]

        print(f"Client {self.client_id}: Phase writer mapping = {phase_writer_mapping}")

        # Create datasets for each phase
        for phase, writers in phase_writer_mapping.items():
            phase_indices = []

            # Find indices in the CLIENT dataset for writers in this phase
            for idx in range(len(self.dataset)):
                # Get the ORIGINAL writer ID for this sample
                original_idx = self.dataset.indices[idx]
                if hasattr(self.dataset.dataset, 'samples'):
                    _, original_label = self.dataset.dataset.samples[original_idx]
                else:
                    _, original_label = self.dataset.dataset[original_idx]

                if original_label in writers:
                    phase_indices.append(idx)

            print(f"Client {self.client_id} Phase {phase}: {len(phase_indices)} samples from writers {writers}")

            # CRITICAL FIX: Use PhaseDataset instead of nested CustomSubset
            phase_dataset = PhaseDataset(self.dataset, phase_indices, writers)
            self.phase_datasets[phase] = phase_dataset

            # Create train/val split
            train_size = int(0.8 * len(phase_dataset))
            val_size = len(phase_dataset) - train_size

            # Create indices for train/val split
            indices = list(range(len(phase_dataset)))
            import random
            random.shuffle(indices)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            # Create data loaders
            train_subset = torch.utils.data.Subset(phase_dataset, train_indices)
            val_subset = torch.utils.data.Subset(phase_dataset, val_indices)

            self.phase_loaders[phase] = {
                'train': DataLoader(train_subset,
                                  batch_size=self.config['client_batch_size'],
                                  shuffle=True,
                                  num_workers=2,
                                  prefetch_factor=2,
                                  pin_memory=True),
                'val': DataLoader(val_subset,
                                batch_size=self.config['client_batch_size'],
                                shuffle=False),
                'writers': writers,  # Original writer IDs
                'train_size': len(train_subset),
                'val_size': len(val_subset),
                'train_indices': train_indices,  # Store for later use
                'val_indices': val_indices
            }

            print(f"  Phase {phase}: Train={len(train_subset)}, Val={len(val_subset)}")

    def update_globally_discovered_writers(self, global_writers):
        """Update the set of globally discovered writers"""
        self.globally_discovered_writers.update(global_writers)
        print(f"Client {self.client_id}: Updated globally discovered writers: {sorted(self.globally_discovered_writers)}")

    def debug_phase_labels(self, phase, num_batches=1):
        """Debug function to check label ranges in a phase"""
        if phase not in self.phase_loaders:
            print(f"Phase {phase} not found")
            return

        print(f"\n🔍 Client {self.client_id} Phase {phase} Label Debug:")
        train_loader = self.phase_loaders[phase]['train']

        batch_count = 0
        for images, labels in train_loader:
            print(f"  Batch {batch_count}: Labels = {labels.unique().tolist()}")
            print(f"  Min label: {labels.min().item()}, Max label: {labels.max().item()}")
            print(f"  Model output size: {self.model.classifier.out_features}")

            batch_count += 1
            if batch_count >= num_batches:
                break

        # Check phase writers
        phase_writers = self.phase_loaders[phase]['writers']
        print(f"  Phase writers (original IDs): {phase_writers}")
        print(f"  Dataset label mapping: {self.dataset.label_mapping}")

    def get_client_update_for_phase(self, phase, global_weights, epochs_per_phase, global_num_classes=None, global_class_registry=None):
        """FIXED: Get complete client update for a specific phase with global class awareness"""
        # Train for the phase with global class information
        local_weights, num_samples, train_loss, train_acc, val_loss, val_acc = self.local_train_phase2(
            global_weights, phase, epochs_per_phase, global_class_registry or {}, global_num_classes
        )

        if local_weights is None:
            return None

        # Get ORIGINAL writer IDs for current phase
        all_phase_writers = self.get_discovered_classes_for_phase(phase)

        # CRITICAL FIX: Filter to only truly NEW discoveries (not globally discovered yet)
        newly_discovered_classes = [w for w in all_phase_writers if w not in self.globally_discovered_writers]

        # Prepare client update with ORIGINAL writer IDs
        client_update = {
            'client_id': self.client_id,
            'weights': local_weights,
            'num_samples': num_samples,
            'discovered_classes': newly_discovered_classes,
            'phase': phase,
            'training_history': self.training_history,
            'performance': {
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            },
            "all_phase_classes": all_phase_writers
        }

        return client_update

    def get_discovered_classes_for_phase(self, phase):
        """Get the ORIGINAL writer IDs discovered in the given phase"""
        if phase in self.phase_loaders:
            return self.phase_loaders[phase]['writers']
        return []

    def predict_writer(self, image):
        """Make prediction and return original writer ID"""
        self.model.eval()
        with torch.no_grad():
            image = image.to(self.device)
            output = self.model(image.unsqueeze(0))
            predicted_class = output.argmax().item()

            # Convert model prediction back to original writer ID
            original_writer_id = self.dataset.get_original_label(predicted_class)
            return original_writer_id

    def update_model_for_new_classes(self, class_mappings):
        """Update local model when new classes are discovered globally"""
        if self.client_id in class_mappings:
            max_class_idx = max(class_mappings[self.client_id].values()) if class_mappings[self.client_id] else 0
            required_classes = max_class_idx + 1

            current_classes = self.model.classifier.out_features
            if required_classes > current_classes:
                print(f"Client {self.client_id}: Expanding model from {current_classes} to {required_classes} classes")
                self.model.expand_classifier(required_classes)

                # CRITICAL: Keep model on device after expansion
                self.model = self.model.to(self.device)
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['local_lr'])

    def debug_device_status(self):
        """Debug function to check all devices"""
        print(f"\n🔍 Client {self.client_id} Device Status:")
        print(f"  Self.device: {self.device}")
        print(f"  Model device: {next(self.model.parameters()).device}")
        print(f"  Criterion device: {self.device}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Current CUDA device: {torch.cuda.current_device()}")

    def get_training_summary(self):
        """Get comprehensive training summary"""
        return {
            'client_id': self.client_id,
            'training_history': self.training_history,
            'total_discovered_writers': len(self.training_history['discovered_writers']),
            'phases_completed': len(self.training_history['phases']),
        }

    def local_train_phase2(self, global_weights, phase, epochs_per_phase, global_class_registry, global_num_classes):
        """
        FIXED: Train locally with a hybrid mapping for known and newly discovered classes.
        Now includes 90-10 train/validation split and validation loop.
        """
        if phase not in self.phase_loaders:
            print(f"⚠️ Client {self.client_id}: Phase {phase} not available")
            return None, 0, 0.0, 0.0, 0.0, 0.0

        print(f"🔍 Client {self.client_id} Phase {phase} - Preparing for training")
        print(f"Client {self.client_id} sees registry: {global_class_registry}")

        # --- START: HYBRID MAP AND MODEL EXPANSION ---

        # 1. CREATE THE HYBRID MAP FOR THIS ROUND
        # Start with the ground truth from the server
        round_training_map = global_class_registry.copy() if global_class_registry else {}
        self.scheduler = StepLR(self.optimizer, step_size=8, gamma=0.8)

        # DEBUG: Print what we received from server
        print(f"  Received global_class_registry: {global_class_registry}")
        print(f"  Round training map initialized: {round_training_map}")

        # Identify writers in this phase that the server doesn't know about yet
        current_phase_writers = set(self.phase_loaders[phase]['writers'])
        known_writers = set(round_training_map.keys())
        newly_discovered_writers = sorted(list(current_phase_writers - known_writers))

        # MINIMAL FIX: Only proceed with new writers if they're actually new
        if newly_discovered_writers:
            print(f"  Client has discovered {len(newly_discovered_writers)} new writers: {newly_discovered_writers}")

            # Get the next available index from the server's perspective
            next_available_idx = max(round_training_map.values()) + 1 if round_training_map else 0

            # Add new writers to the map with PROVISIONAL indices
            for i, new_writer in enumerate(newly_discovered_writers):
                provisional_idx = next_available_idx + i
                round_training_map[new_writer] = provisional_idx
                print(f"  Mapping new writer {new_writer} to provisional index {provisional_idx}")
        else:
            print(f"  No new writers to discover. Using existing mapping: {round_training_map}")

        # 2. EXPAND THE LOCAL MODEL TO FIT THE HYBRID MAP
        # The required size is the total number of classes in our hybrid map
        required_classes = len(round_training_map) if round_training_map else global_num_classes
        current_classes = self.model.classifier.out_features

        # Load global weights BEFORE expanding
        if global_weights:
            self.model.load_state_dict(global_weights)
            self.model = self.model.to(self.device)

        if required_classes > current_classes:
            print(f"  ✅ EXPANDING local model from {current_classes} to {required_classes} classes.")
            self.model.expand_classifier(required_classes)
            self.model = self.model.to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['local_lr'])

        # 3. CREATE DATASET AND SPLIT INTO TRAIN/VAL
        phase_info = self.phase_loaders[phase]
        train_indices = phase_info['train_indices']

        # Instantiate the UPDATED PhaseDataset with the hybrid map
        phase_dataset = PhaseDataset(
            parent_dataset=self.dataset,
            phase_indices=train_indices,
            round_training_map=round_training_map # Pass the complete map
        )

        # Split phase_dataset into 90-10 train/validation
        dataset_size = len(phase_dataset)
        train_size = int(0.9 * dataset_size)
        val_size = dataset_size - train_size

        # Use random_split to create train/val datasets
        from torch.utils.data import random_split
        train_dataset, val_dataset = random_split(
            phase_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducible splits
        )

        print(f"  📊 Dataset split: {train_size} train, {val_size} validation samples")

        # Create train and validation data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['client_batch_size'],
            shuffle=True,
            num_workers=12,
            prefetch_factor=4,
            pin_memory=True,
            persistent_workers=True,
              worker_init_fn=worker_init_fn,
            drop_last=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['client_batch_size'],
            shuffle=False,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True,
            persistent_workers=True,
               worker_init_fn=worker_init_fn,
            drop_last=False
        )

        print(f"🔍 Client {self.client_id} Phase {phase} - Starting training")

        # Debug labels before training
        self.debug_phase_labels(phase, num_batches=1)

        self.current_phase = phase
        self.model.train()

        train_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(epochs_per_phase):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                # Move data to device
                images = images.to(self.device).float()
                labels = labels.to(self.device).long()

                # Debug first batch of first epoch
                if epoch == 0 and batch_idx == 0:
                    print(f"  🔍 First batch debug:")
                    print(f"    Labels: {labels.unique().tolist()}")
                    print(f"    Label range: {labels.min().item()} to {labels.max().item()}")
                    print(f"    Model output size: {self.model.classifier.out_features}")
                    print(f"    Images device: {images.device}")
                    print(f"    Labels device: {labels.device}")
                    print(f"    Model device: {next(self.model.parameters()).device}")

                # Verify labels are in valid range
                if labels.max() >= self.model.classifier.out_features:
                    print(f"⚠️ Client {self.client_id}: Invalid label {labels.max().item()} >= {self.model.classifier.out_features}")
                    print(f"   Skipping batch with invalid labels: {labels.tolist()}")
                    continue

                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += labels.size(0)
                epoch_correct += (predicted == labels).sum().item()

            if len(train_loader) > 0:
                train_loss += epoch_loss / len(train_loader)
                correct += epoch_correct
                total += epoch_total

            # Print progress every few epochs
            if epoch % 5 == 0:
                current_acc = 100. * epoch_correct / epoch_total if epoch_total > 0 else 0.0
                print(f"  Epoch {epoch}: Loss={epoch_loss/len(train_loader):.4f}, Acc={current_acc:.2f}%")
            self.scheduler.step()

        # Calculate training averages
        avg_train_loss = train_loss / epochs_per_phase if epochs_per_phase > 0 else 0.0
        train_accuracy = 100. * correct / total if total > 0 else 0.0

        print(f"Client {self.client_id} Phase {phase} training completed: Loss={avg_train_loss:.4f}, Acc={train_accuracy:.2f}%")

        # --- VALIDATION LOOP ---
        print(f"🔍 Client {self.client_id} Phase {phase} - Starting validation")

        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images = images.to(self.device).float()
                labels = labels.to(self.device).long()

                # Verify labels are in valid range (same check as training)
                if labels.max() >= self.model.classifier.out_features:
                    print(f"⚠️ Client {self.client_id}: Invalid validation label {labels.max().item()} >= {self.model.classifier.out_features}")
                    continue

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate validation averages
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        val_accuracy = 100. * val_correct / val_total if val_total > 0 else 0.0

        print(f"Client {self.client_id} Phase {phase} validation completed: Loss={avg_val_loss:.4f}, Acc={val_accuracy:.2f}%")
        print(f"📊 Final metrics - Train: {train_accuracy:.2f}%, Val: {val_accuracy:.2f}%")

        # Store phase training history (now includes validation metrics)
        self.training_history['phases'][phase] = {
            'train_loss': avg_train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy,
            'writers': list(self.phase_loaders[phase]['writers']),
            'samples': self.phase_loaders[phase]['train_size']
        }

        # Return training metrics (keeping the same interface)
        return self.model.state_dict(), self.phase_loaders[phase]['train_size'], avg_train_loss, train_accuracy, avg_val_loss, val_accuracy

    def evaluate_on_all_previous_classes(self, current_phase):
        """Evaluate model on ALL classes seen up to current phase (excluding current)"""
        if current_phase == self.start_phase:
            return 0.0, 0.0  # No previous classes

        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        batch_count = 0

        # Evaluate on all previous phases
        for phase in range(self.start_phase, current_phase):
            if phase in self.phase_loaders:
                val_loader = self.phase_loaders[phase]['val']

                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(self.device).float()
                        labels = labels.to(self.device).long()

                        if labels.max() >= self.model.classifier.out_features:
                            continue

                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)

                        total_loss += loss.item()
                        batch_count += 1

                        _, predicted = torch.max(outputs, 1)
                        total_samples += labels.size(0)
                        total_correct += (predicted == labels).sum().item()

        if total_samples == 0:
            return 0.0, 0.0

        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        accuracy = 100.0 * total_correct / total_samples

        return avg_loss, accuracy