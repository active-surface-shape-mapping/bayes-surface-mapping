import numpy as np
import torch
from scipy.signal import find_peaks


def find_slice_peak_idxs(
    y_slice,
    *,
    peak_threshold=None,
    max_n_peaks=None,
    peak_height_rule=None,
):
    # One row at a time
    if isinstance(y_slice, torch.Tensor):
        y_np = y_slice.detach().cpu().numpy()
    else:
        y_np = np.asarray(y_slice)

    if y_np.ndim != 1:
        raise ValueError(
            f"find_slice_peak_idxs expected 1D input, got shape {y_np.shape}"
        )

    if max_n_peaks is not None and max_n_peaks <= 0:
        raise ValueError(
            f"max_n_peaks must be positive or None, got {max_n_peaks}"
        )

    valid_rules = {None, "above_median"}
    if peak_height_rule not in valid_rules:
        raise ValueError(
            f"Unknown peak_height_rule={peak_height_rule}. "
            f"Valid options: {valid_rules}"
        )

    # Apply threshold, if provided
    if peak_threshold is not None:
        scale = np.max(y_np) - np.min(y_np)
        prominence = peak_threshold * scale
    else:
        prominence = None

    # Require peaks to be above the slice median, if used
    if peak_height_rule == "above_median":
        height = float(np.median(y_np))
    else:
        height = None

    # Find peaks
    peak_idxs, peak_info = find_peaks(
        y_np,
        prominence=prominence,
        height=height,
    )

    # If max_n_peaks provided, select the max_n_peaks highest peaks
    if max_n_peaks is not None and len(peak_idxs) > max_n_peaks:
        if "peak_heights" in peak_info:
            heights = peak_info["peak_heights"]
        else:
            heights = y_np[peak_idxs]

        ordering = np.argsort(-heights, kind="stable")
        peak_idxs = peak_idxs[ordering][:max_n_peaks]

    return peak_idxs