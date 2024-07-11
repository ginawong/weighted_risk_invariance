import numpy as np
import torch


def compute_ece(predicted, actual, n_bins=10, strategy='quantile'):
    """ Computes expected calibration error (ECE) between predicted and true binary_labels. Implementation works off that
    of calibration_curve from sklearn v1.1.
    Args:
        predicted (torch.Tensor): predicted binary_labels
        actual (torch.Tensor): true binary_labels
        n_bins (int): number of bin_edges to partition probabilities [0, 1] by
        strategy ({'uniform', 'quantile'}): strategy used to identify the widths of the bins
    Returns:
        (float): ECE
    """
    predicted = predicted.detach().squeeze().numpy()
    actual = actual.detach().squeeze().numpy()
    accurate = np.round(predicted) == actual

    if strategy == 'quantile':
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.percentile(predicted, quantiles * 100)
    elif strategy == 'uniform':
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError('Strategy parameter must be either "quantile" or "uniform".')

    # assigns each predicted value to a bin ID (0 to n_bins - 1)
    bin_ids = np.searchsorted(bin_edges[1:-1], predicted)

    # sums the weights assigned to each bin
    # used to compute ECE (computing sums avoids divide-by-0 for empty bins)
    bin_predicted_sum = np.bincount(bin_ids, weights=predicted, minlength=len(bin_edges))
    bin_accurate_count = np.bincount(bin_ids, weights=accurate, minlength=len(bin_edges))

    ece = np.sum(np.abs(bin_accurate_count - bin_predicted_sum)) / len(predicted)

    return float(ece)


def compute_ece_multiclass(predicted, actual, n_bins=10, strategy='quantile'):
    """ Computes expected calibration error (ECE) extended to multi-class
    Args:
        predicted (torch.Tensor): predicted class probabilities (each row should sum to one)
        actual (torch.Tensor): true class labels
        n_bins (int): number of bin_edges to partition probabilities [0, 1] by
        strategy ({'uniform', 'quantile'}): strategy used to identify the widths of the bins
    Returns:
        (float): ECE
    """
    pred, cls = torch.max(predicted, dim=1)
    accurate = (cls == actual).long()
    return compute_ece(pred, accurate, n_bins, strategy)
