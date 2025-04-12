# evaluation.py
"""Functions for generating ground truth and evaluating reconstruction."""

import numpy as np
import logging # Import logging

# --- Get logger for this module ---
logger = logging.getLogger(__name__)
# --- END Get logger ---
# DO NOT call log_config.setup_logging() here


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
    logger.info("Regenerating ground truth baseband for comparison...")
    # Add checks for valid inputs
    if recon_rate is None or recon_rate <= 0 or total_duration < 0:
         logger.error(f"Invalid input for GT generation: duration={total_duration}, recon_rate={recon_rate}")
         return None, None

    num_samples = int(round(total_duration * recon_rate))
    time_vector = np.linspace(0, total_duration, num_samples, endpoint=False) if num_samples > 0 else np.array([])
    gt_baseband = np.zeros(num_samples, dtype=complex)

    mod = global_attrs.get('modulation', 'qam16')
    bw_gt = global_attrs.get('total_signal_bandwidth_hz', None)

    # --- Validation ---
    if bw_gt is None or bw_gt <= 0:
        logger.error(f"Invalid ground truth bandwidth ({bw_gt}) in global_attrs. Cannot regenerate GT.")
        return None, None
    if num_samples <= 0:
        logger.error("Zero samples calculated for GT buffer. Cannot regenerate GT.")
        return None, None
    # --- End Validation ---

    logger.info(f"Attempting GT regeneration for mod: {mod}, BW: {bw_gt/1e6:.2f} MHz, Samples: {num_samples}")
    if mod.lower() == 'qam16':
        try:
            symbol_rate_gt = bw_gt / 4
            if symbol_rate_gt <= 0:
                 logger.error(f"Invalid symbol rate calculated ({symbol_rate_gt}) for QAM16 GT.")
                 return None, None
            logger.info(f"Using GT Symbol Rate = {symbol_rate_gt/1e6:.2f} Msps")
            num_symbols_gt = int(np.ceil(total_duration * symbol_rate_gt))
            if num_symbols_gt > 0:
                logger.debug(f"Generating {num_symbols_gt} QAM16 symbols.")
                symbols = (np.random.choice([-3,-1,1,3], size=num_symbols_gt) + 1j*np.random.choice([-3,-1,1,3], size=num_symbols_gt))/np.sqrt(10)
                samples_per_symbol_gt = recon_rate / symbol_rate_gt
                if samples_per_symbol_gt < 1: logger.warning(f"Samples per symbol < 1 ({samples_per_symbol_gt:.3f})"); samples_per_symbol_gt = 1
                else: samples_per_symbol_gt = int(round(samples_per_symbol_gt))
                logger.debug(f"Using {samples_per_symbol_gt} samples per symbol for GT.")

                baseband_symbols = np.repeat(symbols, samples_per_symbol_gt)
                len_to_take = min(len(baseband_symbols), num_samples)
                gt_baseband[:len_to_take] = baseband_symbols[:len_to_take]
                if len_to_take < num_samples:
                    logger.info(f"GT generated ({len_to_take}) shorter than target buffer ({num_samples}). Padded end with zeros.")
            else:
                logger.warning("Calculated zero GT symbols needed.")
        except Exception as gt_gen_e:
             logger.error(f"Error during QAM16 GT generation: {gt_gen_e}", exc_info=True)
             return None, None
    else:
        logger.warning(f"GT regeneration not implemented for modulation '{mod}'. Returning zeros.")

    # Scale GT to target RMS
    gt_rms_before = np.sqrt(np.mean(np.abs(gt_baseband)**2)) if len(gt_baseband) > 0 else 0.0
    logger.info(f"GT RMS Before Scale: {gt_rms_before:.4e}")
    if gt_rms_before > 1e-20: # Avoid division by zero
        gt_scale_factor = target_rms / gt_rms_before
        gt_baseband *= gt_scale_factor
        gt_rms_after = np.sqrt(np.mean(np.abs(gt_baseband)**2))
        logger.info(f"Scaled GT baseband. Target RMS: {target_rms:.4e}, Actual RMS: {gt_rms_after:.4e}")
    else:
        logger.info("GT baseband power near zero. Scaling skipped.")

    logger.info("Ground truth regeneration complete.")
    return gt_baseband, time_vector


def calculate_metrics(gt_signal, recon_signal, reliable_indices, min_reliable_samples):
    """
    Calculates MSE, NMSE, EVM between ground truth and reconstructed signals using logging.

    Args:
        gt_signal (np.ndarray): Ground truth signal.
        recon_signal (np.ndarray): Final reconstructed signal.
        reliable_indices (np.ndarray): Indices considered reliable (from sum_of_windows).
        min_reliable_samples (int): Minimum number of reliable samples required.

    Returns:
        tuple: (metrics, aligned_recon_signal)
    """
    logger.info("Calculating evaluation metrics using reliable samples...")
    metrics = {'mse': np.inf, 'nmse': np.inf, 'evm': np.inf}
    # Start with the input recon_signal for alignment, default to it if alignment fails
    aligned_recon_signal = recon_signal.copy() if recon_signal is not None else None

    # Basic validation
    if gt_signal is None or recon_signal is None or reliable_indices is None:
        logger.error("Invalid input signals or indices for metric calculation.")
        return metrics, aligned_recon_signal

    max_len_eval = min(len(recon_signal), len(gt_signal))
    if max_len_eval == 0:
         logger.warning("Zero length signals provided for metric calculation.")
         return metrics, aligned_recon_signal

    # Filter reliable indices to be within bounds
    valid_indices_for_eval = reliable_indices[(reliable_indices >= 0) & (reliable_indices < max_len_eval)]
    num_valid_indices = len(valid_indices_for_eval)

    logger.info(f"Using {num_valid_indices} reliable samples (indices < {max_len_eval}) for metrics.")

    if num_valid_indices >= min_reliable_samples:
        gt_reliable = gt_signal[valid_indices_for_eval]
        recon_reliable = recon_signal[valid_indices_for_eval] # Use the FINAL signal passed in

        # Check for finite values within the reliable segments
        if np.all(np.isfinite(gt_reliable)) and np.all(np.isfinite(recon_reliable)):
            mean_power_gt_reliable = np.mean(np.abs(gt_reliable)**2)
            mean_power_recon_reliable = np.mean(np.abs(recon_reliable)**2)
            logger.info(f"  Mean Power GT (Reliable): {mean_power_gt_reliable:.4e}")
            logger.info(f"  Mean Power Recon (Reliable): {mean_power_recon_reliable:.4e}")

            # Calculate metrics if powers are valid
            if mean_power_gt_reliable > 1e-20 and mean_power_recon_reliable > 1e-20:
                power_scale_factor = np.sqrt(mean_power_gt_reliable / mean_power_recon_reliable)
                # Apply alignment scaling to the *entire* recon signal for plotting consistency
                aligned_recon_signal = recon_signal * power_scale_factor
                # Use scaled reliable segment for error calculation
                recon_reliable_scaled = recon_reliable * power_scale_factor
                logger.info(f"  Applied plotting alignment scale factor: {power_scale_factor:.4f}")

                error_reliable = gt_reliable - recon_reliable_scaled
                mse = np.mean(np.abs(error_reliable)**2)
                nmse = mse / mean_power_gt_reliable
                if nmse >= 0:
                    evm = np.sqrt(nmse) * 100
                else:
                    logger.warning(f"Calculated negative NMSE ({nmse:.4e}). Setting EVM to Inf.")
                    evm = np.inf

                metrics['mse'] = mse
                metrics['nmse'] = nmse
                metrics['evm'] = evm
                # Log the RMS of the signal used for plotting
                rms_aligned_check = np.sqrt(np.mean(np.abs(aligned_recon_signal)**2))
                logger.info(f"  RMS of Aligned Recon (for plotting): {rms_aligned_check:.4e}")
            else:
                logger.warning("Near-zero power in reliable segments. Cannot calculate metrics/align.")
                # aligned_recon_signal remains the unaligned input recon_signal in this case
        else:
            logger.warning("Non-finite values found in reliable segments. Cannot calculate metrics.")
            # aligned_recon_signal remains the unaligned input recon_signal
    else:
        logger.warning(f"Not enough reliable samples ({num_valid_indices}) for evaluation (Min {min_reliable_samples} required).")
        # aligned_recon_signal remains the unaligned input recon_signal

    logger.info("--- Evaluation Metrics Results ---")
    logger.info(f"  MSE : {metrics['mse']:.4e}")
    nmse_val = metrics['nmse']
    if np.isfinite(nmse_val):
        if nmse_val > 1e-20: # Check for meaningful positive value before log10
            logger.info(f"  NMSE: {nmse_val:.4e} ({10*np.log10(nmse_val):.2f} dB)")
        else:
            logger.info(f"  NMSE: {nmse_val:.4e} (dB value too low or invalid)")
    else:
        logger.info(f"  NMSE: {nmse_val}") # Print Inf/NaN directly
    logger.info(f"  EVM : {metrics['evm']:.2f}%" if np.isfinite(metrics['evm']) else f"EVM : {metrics['evm']}")
    logger.info("---------------------------------")


    return metrics, aligned_recon_signal