 

def create_train_test_split(source_path, output_base_path, split_ratio=0.8, seed=42):
    """
    Split dataset physically into train/test directories

    Args:
        source_path: Path to original dataset (ImageFolder structure)
        output_base_path: Base path where train/test folders will be created
        split_ratio: Ratio for train split (0.8 = 80% train, 20% test)
        seed: Random seed for reproducible splits
    """
    random.seed(seed)

    source_path = Path(source_path)
    output_base_path = Path(output_base_path)

    # Create train and test directories
    train_path = output_base_path / "train"
    test_path = output_base_path / "test"

    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    print(f"Splitting {source_path} into:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")

    total_train_samples = 0
    total_test_samples = 0

    # Process each class directory
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]

    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"Processing class: {class_name}")

        # Create class directories in train and test
        train_class_dir = train_path / class_name
        test_class_dir = test_path / class_name

        train_class_dir.mkdir(exist_ok=True)
        test_class_dir.mkdir(exist_ok=True)

        # Get all files in this class
        image_files = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]

        # Shuffle files for random split
        random.shuffle(image_files)

        # Calculate split point
        train_count = int(len(image_files) * split_ratio)

        train_files = image_files[:train_count]
        test_files = image_files[train_count:]

        # Copy files to train directory
        for file in train_files:
            shutil.copy2(file, train_class_dir / file.name)

        # Copy files to test directory
        for file in test_files:
            shutil.copy2(file, test_class_dir / file.name)

        total_train_samples += len(train_files)
        total_test_samples += len(test_files)

        print(f"  {class_name}: {len(train_files)} train, {len(test_files)} test")

    print(f"\n✅ Split complete!")
    print(f"Total train samples: {total_train_samples}")
    print(f"Total test samples: {total_test_samples}")
    print(f"Split ratio: {total_train_samples/(total_train_samples + total_test_samples):.3f}")

    return train_path, test_path

def create_client_datasets(dataset, client_writer_mapping):
    """Create client datasets with proper label remapping"""
    client_datasets = {}
    for client_id, writer_classes in client_writer_mapping.items():
        client_indices = []
        writer_ids = set()

        # Collect indices for this client's writers
        for idx, (_, label) in enumerate(dataset.samples):
            if label in writer_classes:
                client_indices.append( idx)
                writer_ids.add(label)

        print(f"Client {client_id} writers: {sorted(writer_ids)}")

        # Create CustomSubset with automatic label remapping
        client_dataset = CustomSubset(dataset, client_indices)
        client_datasets[client_id] = client_dataset

        print(f"Client {client_id} label mapping: {client_dataset.label_mapping}")

    return client_datasets

