from net.evaluation.dMean import d_mean_mm

def compute_metrics(outputs, targets, spacings):
    """
    Computes and optionally aggregates evaluation metrics.

    Args:
        outputs (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth values.
        spacings (torch.Tensor): Spacing values for distance calculations.
        exp_dir (str): Experiment directory (not used in function but kept for flexibility).
        metrics_list (list, optional): List of previously computed metrics for aggregation.

    Returns:
        dict: Scores.
    """
    # Define metric functions
    metric_functions = {
        "dmean": d_mean_mm,
    }

    # Compute metrics
    scores = {key: func(outputs, targets, spacings) for key, func in metric_functions.items()}

    return scores

def compute_landmark_metrics(outputs, targets, spacings):
    """
    Computes and optionally aggregates evaluation metrics.

    Args:
        outputs (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth values.
        spacings (torch.Tensor): Spacing values for distance calculations.
        exp_dir (str): Experiment directory (not used in function but kept for flexibility).
        metrics_list (list, optional): List of previously computed metrics for aggregation.

    Returns:
        dict: Scores.
    """
    # Define metric functions
    metric_functions = {
        "dmean": d_mean_mm,
    }
    # Compute metrics
    scores = {key: func(outputs, targets, spacings) for key, func in metric_functions.items()}
    
    return scores

def compute_heatmap_metrics(outputs, targets):
    """
    Computes and optionally aggregates evaluation metrics.

    Args:
        outputs (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth values.

    Returns:
        dict: Scores.
    """
    # Define metric functions
    metric_functions = {
         # corr 
         # r2  
         # mae
    }

    # Compute metrics
    scores = {key: func(outputs, targets) for key, func in metric_functions.items()}

    return scores
