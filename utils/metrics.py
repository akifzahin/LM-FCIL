def calculate_comprehensive_metrics(server, test_loader, device='cuda'):
    """
    Calculate comprehensive metrics using the SAME logic as server.global_evaluate()
    Only evaluates on samples whose raw writer-ID is registered and remaps labels.
    """
    server.global_model.eval()

    all_predictions = []
    all_labels = []
    all_probabilities = []

    total_inference_time = 0
    total_samples = 0
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, raw_labels in test_loader:
            batch_size = raw_labels.size(0)
            Cg = server.global_model.classifier.out_features

            # Build mask on CPU: which raw_labels are in class_registry? (SAME AS YOUR CODE)
            mask_cpu = torch.zeros(batch_size, dtype=torch.bool)
            mapped_cpu = torch.zeros(batch_size, dtype=torch.long)

            for i, raw in enumerate(raw_labels.tolist()):
                if raw in server.class_registry:
                    mask_cpu[i] = True
                    mapped_cpu[i] = server.class_registry[raw]

            if mask_cpu.sum().item() == 0:
                # No registered writers in this batch → skip (SAME AS YOUR CODE)
                continue

            # Keep only the masked samples (still on CPU) (SAME AS YOUR CODE)
            imgs_cpu = images[mask_cpu]
            targets_cpu = mapped_cpu[mask_cpu]

            # Now move sliced tensors to device (SAME AS YOUR CODE)
            imgs = imgs_cpu.to(device).float()
            targets = targets_cpu.to(device).long()

            # Measure inference time
            start_time = time.time()
            outputs = server.global_model(imgs)
            inference_time = time.time() - start_time

            total_inference_time += inference_time

            # Calculate loss (SAME AS YOUR CODE)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            total_loss += loss.item() * imgs.size(0)

            # Get probabilities and predictions
            probabilities = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # Accumulate for metrics calculation
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            # Count correct predictions (SAME AS YOUR CODE)
            correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

    if total_samples == 0:
        return {
            'accuracy': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_macro': 0.0,
            'precision_weighted': 0.0,
            'recall_weighted': 0.0,
            'f1_weighted': 0.0,
            'auc_roc': 0.0,
            'inference_time_per_sample_ms': 0.0,
            'total_inference_time_s': 0.0,
            'total_samples': 0,
            'loss': float('inf')
        }

    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)

    # Calculate accuracy (should match your server.global_evaluate exactly)
    accuracy = correct / total_samples  # This should give you 0.88x
    avg_loss = total_loss / total_samples

    # Calculate precision, recall, F1 (macro and weighted averages)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    # Calculate AUC-ROC
    try:
        num_classes = len(np.unique(y_true))
        if num_classes == 2:
            auc_roc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            # Multi-class: use label_binarize with the actual unique classes present
            unique_classes = np.unique(y_true)
            y_true_binarized = label_binarize(y_true, classes=unique_classes)
            if y_true_binarized.shape[1] == 1:
                auc_roc = 0.5
            else:
                # Only use the columns corresponding to classes that are actually present
                y_prob_subset = y_prob[:, unique_classes]
                auc_roc = roc_auc_score(y_true_binarized, y_prob_subset, multi_class='ovr', average='macro')
    except Exception as e:
        print(f"Warning: Could not calculate AUC-ROC: {e}")
        auc_roc = 0.0

    # Calculate average inference time per sample
    avg_inference_time_per_sample = total_inference_time / total_samples

    return {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'auc_roc': float(auc_roc),
        'inference_time_per_sample_ms': float(avg_inference_time_per_sample * 1000),
        'total_inference_time_s': float(total_inference_time),
        'total_samples': int(total_samples),
        'loss': float(avg_loss)
    }

from datetime import datetime
results = {
    'experiment_info': {
        'model': template,  # Use your template variable instead of hardcoding
        'timestamp': datetime.now().isoformat(),
        'num_clients': len(clients),
        'total_writers': len(server.class_registry) if hasattr(server, 'class_registry') else 0,
        'phases': FL_CONFIG['phases'],
        'best_accuracy': best_val_acc
    },
    'global_accuracy_progression': accuracies_for_later,  # Your existing list
    'phase_comprehensive_metrics': make_json_serializable(phase_comprehensive_metrics),  # NEW: Comprehensive metrics
    'client_summaries': {}
}
num_classes_evaluated = { 1: 12, 2: 15, 3: 18}
# Save individual client summaries
for client_id, client in clients.items():
    summary = client.get_training_summary()
    results['client_summaries'][client_id] = make_json_serializable(summary)

# Save to JSON file with comprehensive metrics
filename = f"federated_results_{template.lower()}1.json"
with open(filename, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Enhanced results with comprehensive metrics saved to {filename}")

# Create comprehensive metrics CSV for easy analysis
comprehensive_df_data = []
for phase, phase_data in phase_comprehensive_metrics.items():
    metrics = phase_data['metrics']
    comprehensive_df_data.append({
        'Phase': phase,
        'Round': phase_data['round'],
        'Accuracy': metrics['accuracy'],
        'F1_Macro': metrics['f1_macro'],
        'F1_Weighted': metrics['f1_weighted'],
        'Precision_Macro': metrics['precision_macro'],
        'Recall_Macro': metrics['recall_macro'],
        'AUC_ROC': metrics['auc_roc'],
        'Inference_Time_ms': metrics['inference_time_per_sample_ms'],
        'Loss': metrics['loss'],
         'Num_Classes':num_classes_evaluated
    })

if comprehensive_df_data:
    comprehensive_df = pd.DataFrame(comprehensive_df_data)
    comprehensive_csv_filename = f"comprehensive_metrics_{template.lower()}1.csv"
    comprehensive_df.to_csv(comprehensive_csv_filename, index=False)
    print(f"Comprehensive metrics CSV saved to {comprehensive_csv_filename}")

# Also keep your existing CSV for backward compatibility
df_data = []
for phase_data in accuracies:
    phase, round_num, accuracy = phase_data
    df_data.append({
        'Phase': phase,
        'Round': round_num,
        'Global_Accuracy': accuracy
    })

df = pd.DataFrame(df_data)
csv_filename = f"global_accuracy_{template.lower()}1.csv"
df.to_csv(csv_filename, index=False)
print(f"Global accuracy CSV saved to {csv_filename}")



print(f"\n📊 COMPREHENSIVE METRICS SUMMARY:")
print(f"{'='*60}")
for phase, phase_data in phase_comprehensive_metrics.items():
    metrics = phase_data['metrics']
    print(f"\nPhase {phase} Final Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-Score (Macro/Weighted): {metrics['f1_macro']:.4f} / {metrics['f1_weighted']:.4f}")
    print(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro): {metrics['recall_macro']:.4f}")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  Inference Time: {metrics['inference_time_per_sample_ms']:.4f} ms/sample")
    print(f"  Classes: {num_classes_evaluated}")