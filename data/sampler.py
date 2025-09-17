from torch.utils.data import WeightedRandomSampler

def create_balanced_sampler(subset):
    """
    Create a sampler that balances classes for any Subset-like dataset whose
    underlying dataset remaps labels (e.g. PhaseDataset).

    Args:
        subset: a torch.utils.data.Subset or PhaseDataset directly
    Returns:
        WeightedRandomSampler that will yield each global class equally often
    """
    # Determine where to pull labels from:
    # - If this is a Subset, unwrap to get the PhaseDataset
    base = subset
    indices = None
    if isinstance(subset, torch.utils.data.Subset):
        base = subset.dataset
        indices = subset.indices
    else:
        # If it's a PhaseDataset itself, it has phase_indices
        indices = list(range(len(base)))

    # Now collect the remapped labels for each sample in this subset
    remapped_labels = []
    for local_idx in indices:
        # __getitem__ of PhaseDataset returns (image, new_label)
        _, lbl = base[local_idx]
        remapped_labels.append(int(lbl))

    # Count frequency of each class in this subset
    from collections import Counter
    class_counts = Counter(remapped_labels)

    # Build per-sample weights = inverse class frequency
    weights = [1.0 / class_counts[lbl] for lbl in remapped_labels]

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
