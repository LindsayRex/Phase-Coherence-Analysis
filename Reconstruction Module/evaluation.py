# evaluation.py
"""Functions for generating ground truth and evaluating reconstruction."""

import numpy as np
import logging
from . import log_config

# Setup logging for this module
log_config.setup_logging(level=logging.DEBUG, log_dir="run_logs")
logger = logging.getLogger(__name__)

def regenerate_ground_truth(global_attrs, total_duration, recon_rate, target_rms):
    """
    Regenerates the ground truth baseband signal based on global attributes.

    Args:
        global_attrs (dict): Dictionary of global parameters from HDF5.
        total_duration (float): The total duration of the reconstructed signal (seconds).
        recon_rate (float): The reconstruction sample rate (Hz).
        target_rms (float): The target RMS for the GT signal.

    Returns:
        tuple: (gt_baseband, time_vector)
               - gt_baseband (np.ndarray): The generated ground truth signal (complex128).
               - time_vector (np.ndarray): Corresponding time vector.
               Returns (None, None) if generation fails.
    """
    logger.info("\nRegenerating ground truth baseband for comparison...")
    num_samples = int(round(total_duration * recon_rate)) if recon_rate is not None and total_duration >= 0 else 0
    time_vector = np.linspace(0, total_duration, num_samples, endpoint=False) if num_samples > 0 else np.array([])
    gt_baseband = np.zeros(num_samples, dtype=complex)

    mod = global_attrs.get('modulation', 'qam16')
    bw_gt = global_attrs.get('total_signal_bandwidth_hz', None)

    # --- Validation ---
    if bw_gt is None or bw_gt <= 0:
        logger.error(f"Invalid ground truth bandwidth ({bw_gt}). Cannot regenerate GT.")
        return None, None
    if recon_rate is None or recon_rate <= 0:
        logger.error(f"Invalid reconstruction sample rate ({recon_rate}). Cannot regenerate GT.")
        return None, None
    if num_samples <= 0:
        logger.error("Zero samples calculated for GT buffer.")
        return None, None
    # --- End Validation ---

    logger.info(f"Attempting GT regeneration for mod: {mod}, BW: {bw_gt/1e6:.2f} MHz")
    if mod.lower() == 'qam16':
        try:
            symbol_rate_gt = bw_gt / 4
            logger.info(f"Using GT Symbol Rate = {symbol_rate_gt/1e6:.2f} Msps")
            num_symbols_gt = int(np.ceil(total_duration * symbol_rate_gt))
            if num_symbols_gt > 0:
                symbols = (np.random.choice([-3,-1,1,3], size=num_symbols_gt) + 1j*np.random.choice([-3,-1,1,3], size=num_symbols_gt))/np.sqrt(10)
                samples_per_symbol_gt = max(1, int(round(recon_rate/symbol_rate_gt)))
                baseband_symbols = np.repeat(symbols, samples_per_symbol_gt)
                len_to_take = min(len(baseband_symbols), num_samples)
                gt_baseband[:len_to_take] = baseband_symbols[:len_to_take]
                if len_to_take < num_samples:
                    logger.info(f"GT generated ({len_to_take}) shorter than target ({num_samples}). Padded.")
            else:
                logger.warning("Calculated zero GT symbols.")
        except Exception as gt_gen_e:
            logger.error(f"Error during QAM16 GT generation: {gt_gen_e}")
            return None, None
    else:
        logger.warning(f"GT regeneration not implemented for '{mod}'. Returning zeros.")

    # Scale GT to target RMS
    gt_rms_before = np.sqrt(np.mean(np.abs(gt_baseband)**2))
    logger.info(f"GT RMS Before Scale: {gt_rms_before:.4e}")
    if gt_rms_before > 1e-20:
        gt_scale_factor = target_rms / gt_rms_before
        gt_baseband *= gt_scale_factor
        gt_rms_after = np.sqrt(np.mean(np.abs(gt_baseband)**2))
        logger.info(f"Scaled GT baseband. Target RMS: {target_rms:.4e}, Actual RMS: {gt_rms_after:.4e}")
    else:
        logger.info("GT baseband power near zero. Scaling skipped.")

    return gt_baseband, time_vector

def calculate_metrics(gt_signal, recon_signal, reliable_indices, min_reliable_samples):
    """
    Calculates MSE, NMSE, EVM between ground truth and reconstructed signals.

    Args:
        gt_signal (np.ndarray): Ground truth signal.
        recon_signal (np.ndarray): Final reconstructed signal.
        reliable_indices (np.ndarray): Indices considered reliable (from sum_of_windows).
        min_reliable_samples (int): Minimum number of reliable samples required.

    Returns:
        tuple: (metrics, aligned_recon_signal)
               - metrics (dict): {'mse': float, 'nmse': float, 'evm': float}. Values are np.inf if calculation fails.
               - aligned_recon_signal (np.ndarray): Reconstructed signal scaled for plotting alignment.
    """
    logger.info("\nCalculating metrics using reliable samples.")
    metrics = {'mse': np.inf, 'nmse': np.inf, 'evm': np.inf}
    aligned_recon_signal = recon_signal.copy()

    max_len_eval = min(len(recon_signal), len(gt_signal))
    valid_indices_for_eval = reliable_indices[(reliable_indices >= 0) & (reliable_indices < max_len_eval)]

    logger.info(f"Using {len(valid_indices_for_eval)} reliable samples (indices < {max_len_eval}).")
    if len(valid_indices_for_eval) >= min_reliable_samples:
        gt_reliable = gt_signal[valid_indices_for_eval]
        recon_reliable = recon_signal[valid_indices_for_eval]

        if np.all(np.isfinite(gt_reliable)) and np.all(np.isfinite(recon_reliable)):
            mean_power_gt_reliable = np.mean(np.abs(gt_reliable)**2)
            mean_power_recon_reliable = np.mean(np.abs(recon_reliable)**2)
            logger.info(f"Mean Power GT (Reliable): {mean_power_gt_reliable:.4e}")
            logger.info(f"Mean Power Recon (Reliable): {mean_power_recon_reliable:.4e}")

            if mean_power_gt_reliable > 1e-20 and mean_power_recon_reliable > 1e-20:
                power_scale_factor = np.sqrt(mean_power_gt_reliable / mean_power_recon_reliable)
                aligned_recon_signal = recon_signal * power_scale_factor
                recon_reliable_scaled = recon_reliable * power_scale_factor
                logger.info(f"Applied plotting alignment scale factor: {power_scale_factor:.4f}")

                error_reliable = gt_reliable - recon_reliable_scaled
                mse = np.mean(np.abs(error_reliable)**2)
                nmse = mse / mean_power_gt_reliable
                evm = np.sqrt(nmse) * 100 if nmse >= 0 else np.inf

                metrics['mse'] = mse
                metrics['nmse'] = nmse
                metrics['evm'] = evm
                rms_aligned_check = np.sqrt(np.mean(np.abs(aligned_recon_signal)**2))
                logger.info(f"RMS of Aligned Recon (for plotting): {rms_aligned_check:.4e}")
            else:
                logger.warning("Near-zero power in reliable segments. Cannot calculate metrics/align.")
        else:
            logger.warning("Non-finite values in reliable segments. Cannot calculate metrics.")
    else:
        logger.warning(f"Not enough reliable samples ({len(valid_indices_for_eval)}) for evaluation (Min {min_reliable_samples} required).")

    logger.info("\nEvaluation Metrics:")
    logger.info(f"MSE : {metrics['mse']:.4e}")
    nmse_val = metrics['nmse']
    if np.isfinite(nmse_val) and nmse_val > 1e-20:
        logger.info(f"NMSE: {nmse_val:.4e} ({10*np.log10(nmse_val):.2f} dB)")
    elif np.isfinite(nmse_val):
        logger.info(f"NMSE: {nmse_val:.4e} (dB invalid)")
    else:
        logger.info(f"NMSE: {nmse_val}")
    logger.info(f"EVM : {metrics['evm']:.2f}%" if np.isfinite(metrics['evm']) else f"EVM : {metrics['evm']}")

    return metrics, aligned_recon_signal