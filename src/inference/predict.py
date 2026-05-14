import torch
from torch.utils.data import DataLoader


def predict_proba(model, dataset, device, batch_size=32):
    """
    Run inference and return record-level probabilities.

    For ECGDataset (windowed CNN): applies sigmoid per window, then
    mean-pools probabilities across windows belonging to the same record.
    For ECGDatasetFull (transformers): one record = one sample, returned as-is.

    Args:
        model:      nn.Module in eval mode
        dataset:    ECGDataset or ECGDatasetFull instance
        device:     torch.device
        batch_size: DataLoader batch size

    Returns:
        probs:  (N_records, 5) float tensor -- sigmoid probabilities
        labels: (N_records, 5) float tensor -- ground-truth multi-hot
    """
    from src.data.dataset import ECGDataset

    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_probs, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(y)

    all_probs  = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    if isinstance(dataset, ECGDataset):
        n_records  = len(dataset.df)
        rec_probs  = torch.zeros(n_records, all_probs.shape[1])
        rec_labels = torch.zeros(n_records, all_probs.shape[1])
        counts     = torch.zeros(n_records)
        for i, (rec_idx, _) in enumerate(dataset.index):
            rec_probs[rec_idx]  += all_probs[i]
            rec_labels[rec_idx]  = all_labels[i]
            counts[rec_idx]     += 1
        all_probs  = rec_probs / counts.unsqueeze(1)
        all_labels = rec_labels

    return all_probs, all_labels
