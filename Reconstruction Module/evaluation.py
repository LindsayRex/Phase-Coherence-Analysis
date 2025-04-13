# evaluation.py
"""Functions for evaluating reconstruction quality."""

import numpy as np
import logging # Import standard logging

# --- Get logger for this module ---
logger = logging.getLogger(__name__)
# --- End Get logger ---
# DO NOT call log_config.setup_logging() here

# --- Function regenerate_ground_truth removed ---
# (This function is no longer needed as GT is loaded from HDF5 and resampled in main.py)


def calculate_metrics(gt_signal_eval_rate, recon_signal_eval_rate, reliable_indices, min_reliable_samples):
    """
    Calculates MSE, NMSE, EVM between ground truth and reconstructed signals.
    Assumes both input signals are at the same final evaluation sample rate.

    Args:
        gt_signal_eval_rate (np.ndarray): Ground truth signal at the final evaluation rate.
        recon_signal_eval_rate (np.ndarray): Final reconstructed signal at the evaluation rate.
        reliable_indices (np.ndarray): Indices considered reliable (e.g., from sum_of_windows).
        min_reliable_samples (int): Minimum number of reliable samples required for metrics.

    Returns:
        tuple: (metrics, aligned_recon_signal)
               - metrics (dict): {'mse': float, 'nmse': float, 'evm': float}. Values are np.inf if calculation fails.
               - aligned_recon_signal (np.ndarray): Reconstructed signal scaled for plotting alignment.
    """
    logger.info("--- Calculating Evaluation Metrics ---")
    metrics = {'mse': np.inf, 'nmse': np.inf, 'evm': np.inf}
    # Start with the input recon_signal for alignment, return it if alignment fails
    aligned_recon_signal = recon_signal_eval_rate.copy() if recon_signal_eval_rate is not None else None

    # Basic validation
    if gt_signal_eval_rate is None: logger.error("GT signal is None. Cannot calculate metrics."); return metrics, aligned_recon_signal
    if recon_signal_eval_rate is None: logger.error("Recon signal is None. Cannot calculate metrics."); return metrics, aligned_recon_signal
    if reliable_indices is None: logger.warning("Reliable indices is None. Proceeding using all samples."); reliable_indices = np.arange(len(recon_signal_eval_rate)) # Fallback to all indices

    # Ensure signals have the same length
    if len(gt_signal_eval_rate) != len(recon_signal_eval_rate):
        logger.error(f"Length mismatch for metrics: GT={len(gt_signal_eval_rate)}, Recon={len(recon_signal_eval_rate)}")
        return metrics, aligned_recon_signal

    max_len_eval = len(recon_signal_eval_rate)
    if max_len_eval == 0: logger.warning("Zero length signals provided."); return metrics, aligned_recon_signal

    # Filter reliable indices to be within bounds
    valid_indices_for_eval = reliable_indices[(reliable_indices >= 0) & (reliable_indices < max_len_eval)]
    num_valid_indices = len(valid_indices_for_eval)

    logger.info(f"Using {num_valid_indices} reliable samples (indices < {max_len_eval}) for metrics.")

    if num_valid_indices >= min_reliable_samples:
        # Select reliable segments
        gt_reliable = gt_signal_eval_rate[valid_indices_for_eval]
        recon_reliable = recon_signal_eval_rate[valid_indices_for_eval] # Use the signal passed in

        # Check for finite values
        if np.all(np.isfinite(gt_reliable)) and np.all(np.isfinite(recon_reliable)):
            mean_power_gt_reliable = np.mean(np.abs(gt_reliable)**2)
            mean_power_recon_reliable = np.mean(np.abs(recon_reliable)**2)
            logger.info(f"  Mean Power GT (Reliable Segment): {mean_power_gt_reliable:.4e}")
            logger.info(f"  Mean Power Recon (Reliable Segment): {mean_power_recon_reliable:.4e}")

            # Calculate metrics if powers are valid
            if mean_power_gt_reliable > 1e-20 and mean_power_recon_reliable > 1e-20:
                # Calculate scale factor based *only* on reliable segment powers
                power_scale_factor = np.sqrt(mean_power_gt_reliable / mean_power_recon_reliable)
                # Apply alignment scaling to the *entire* recon signal for plotting
                aligned_recon_signal = recon_signal_eval_rate * power_scale_factor
                # Use the scaled reliable segment for error calculation
                recon_reliable_scaled = recon_reliable * power_scale_factor
                logger.info(f"  Applied plotting alignment scale factor: {power_scale_factor:.4f}")

                error_reliable = gt_reliable - recon_reliable_scaled
                mse = np.mean(np.abs(error_reliable)**2)
                nmse = mse / mean_power_gt_reliable
                if nmse >= 0:
                    evm = np.sqrt(nmse) * 100
                else:
                    logger.warning(f"Calculated negative NMSE ({nmse:.4e}). EVM set to Inf.")
                    evm = np.inf

                metrics['mse'] = mse; metrics['nmse'] = nmse; metrics['evm'] = evm
                rms_aligned_check = np.sqrt(np.mean(np.abs(aligned_recon_signal)**2))
                logger.info(f"  RMS of Aligned Recon (for plotting): {rms_aligned_check:.4e}")
            else:
                logger.warning("Near-zero power in reliable segments. Cannot calculate metrics/align.")
        else:
            logger.warning("Non-finite values found in reliable segments. Cannot calculate metrics.")
    else:
        logger.warning(f"Not enough reliable samples ({num_valid_indices}) for evaluation (Min {min_reliable_samples} required). Metrics remain Inf.")

    logger.info("--- Evaluation Metrics Results ---")
    logger.info(f"  MSE : {metrics['mse']:.4e}")
    nmse_val = metrics['nmse']
    if np.isfinite(nmse_val):
        if nmse_val > 1e-20: logger.info(f"  NMSE: {nmse_val:.4e} ({10*np.log10(nmse_val):.2f} dB)")
        else: logger.info(f"  NMSE: {nmse_val:.4e} (dB value too low or invalid)")
    else: logger.info(f"  NMSE: {nmse_val}")
    logger.info(f"  EVM : {metrics['evm']:.2f}%" if np.isfinite(metrics['evm']) else f"EVM : {metrics['evm']}")
    logger.info("---------------------------------")

    # Return calculated metrics and the aligned signal (which might be the original if alignment failed)
    return metrics, aligned_recon_signal