# stitching.py
"""Functions for stitching chunks together using overlap-add."""

import numpy as np
from scipy import signal as sig
from tqdm import tqdm
import logging
from . import log_config

# Setup logging for this module
log_config.setup_logging(level=logging.DEBUG, log_dir="run_logs")
logger = logging.getLogger(__name__)

def normalize_chunks_pre_stitch(chunks, target_rms):
    """Normalizes each chunk individually to the target RMS before stitching."""
    logger.info("\n--- Normalizing chunks before stitching ---")
    normalized_chunks = []
    logger.info("Chunk | RMS Before Norm | RMS After Norm")
    logger.info("------|-----------------|---------------")
    for i, chunk in enumerate(chunks):
        rms_before = np.nan
        rms_after = np.nan
        norm_chunk = chunk

        if len(chunk) > 0 and np.all(np.isfinite(chunk)):
            chunk_rms = np.sqrt(np.mean(np.abs(chunk)**2))
            rms_before = chunk_rms
            if chunk_rms > 1e-12:
                scale = target_rms / chunk_rms
                norm_chunk = chunk * scale
                rms_after = np.sqrt(np.mean(np.abs(norm_chunk)**2)) # Verify RMS
            else:
                norm_chunk = np.zeros_like(chunk)
                rms_after = 0.0
        elif len(chunk) == 0:
             norm_chunk = chunk; rms_before = 0.0; rms_after = 0.0
        else: # Non-finite
             logger.warning(f"Warning: Chunk {i} contains non-finite values before norm. Zeroing.")
             norm_chunk = np.zeros_like(chunk); rms_after = 0.0

        normalized_chunks.append(norm_chunk)
        logger.info(f"{i:<5d} | {rms_before:<15.4e} | {rms_after:.4e}")
    logger.info("--- Normalization Complete ---")
    return normalized_chunks


def stitch_signal(chunks, sdr_rate, recon_rate, overlap_factor, tuning_delay, window_type):
    """
    Stitches chunks using overlap-add method with specified window.

    Args:
        chunks (list): List of normalized chunks (complex128 numpy arrays).
        sdr_rate (float): Original SDR sample rate (Hz). Used for duration/advance.
        recon_rate (float): Sample rate of the input chunks (Hz).
        overlap_factor (float): The effective overlap factor used.
        tuning_delay (float): Tuning delay between chunks (seconds).
        window_type (str): Type of window for overlap-add (e.g., 'blackmanharris').

    Returns:
        tuple: (reconstructed_signal, sum_of_windows)
               - reconstructed_signal (np.ndarray): Raw stitched signal before final normalization.
               - sum_of_windows (np.ndarray): Sum of window functions used (for normalization).
        Returns (None, None) if stitching cannot be performed.
    """
    logger.info("\n--- Performing Stitching with Improved Windows ---")
    if not chunks:
        logger.error("Error: No chunks provided for stitching.")
        return None, None

    # Calculate buffer size and timing
    chunk_duration_s = 0
    if len(chunks[0]) > 0 and recon_rate > 0: # Estimate from first chunk
        chunk_duration_s = len(chunks[0]) / recon_rate
    elif sdr_rate > 0 and len(chunks[0]) > 0: # Fallback to SDR rate if recon rate failed? Less accurate.
         # This assumes chunk lengths correspond to original SDR capture time, which might not be true after upsampling.
         # Best to ensure recon_rate is valid.
         logger.warning("Warning: Estimating chunk duration from SDR rate for stitching.")
         chunk_duration_s = len(chunks[0]) / sdr_rate # This is likely incorrect if chunks are upsampled
         # Better: Recalculate duration based on original chunk length before upsampling if possible

    if chunk_duration_s <= 0:
         logger.error("Error: Cannot determine chunk duration for stitching buffer calculation.")
         return None, None

    time_advance_per_chunk = chunk_duration_s * (1.0 - overlap_factor)
    # Use max(0, len(chunks)-1) in case there's only one chunk
    total_duration_recon = chunk_duration_s + max(0, len(chunks) - 1) * (time_advance_per_chunk + tuning_delay)
    num_samples_recon = int(round(total_duration_recon * recon_rate))

    if num_samples_recon <= 0:
        logger.error(f"Error: Invalid calculated reconstruction samples ({num_samples_recon}).")
        return None, None

    reconstructed_signal = np.zeros(num_samples_recon, dtype=complex)
    sum_of_windows = np.zeros(num_samples_recon, dtype=float)

    logger.info(f"Reconstruction target buffer: {num_samples_recon} samples @ {recon_rate/1e6:.2f} MHz (Est. Duration: {total_duration_recon*1e6:.1f} us)")
    logger.info(f"Using stitching window: '{window_type}' with effective overlap factor: {overlap_factor:.2f}")

    current_recon_time_start = 0.0

    for i, chunk_to_add in tqdm(enumerate(chunks), total=len(chunks), desc="Stitching"):
        start_idx_recon = int(round(current_recon_time_start * recon_rate))
        num_samples_in_chunk = len(chunk_to_add)
        end_idx_recon = min(start_idx_recon + num_samples_in_chunk, num_samples_recon)
        actual_len = end_idx_recon - start_idx_recon

        if actual_len <= 0 or num_samples_in_chunk == 0:
            if i < len(chunks) - 1: current_recon_time_start += time_advance_per_chunk + tuning_delay
            continue

        window = np.ones(actual_len)
        if actual_len >= 2:
            try: window = sig.get_window(window_type, actual_len)
            except: window = np.ones(actual_len); logger.warning(f"Warn: Failed get_window '{window_type}'")

        # Window for sum_of_windows divisor (RMS=1 normalization)
        window_rms = np.sqrt(np.mean(window**2))
        normalized_window_for_sum = np.ones_like(window)
        if window_rms > 1e-9: normalized_window_for_sum = window / window_rms

        try:
            segment_data = chunk_to_add[:actual_len]
            if not np.all(np.isfinite(segment_data)):
                segment_data = np.nan_to_num(segment_data); logger.warning(f"Warn: NaN in chunk {i} stitch")
            windowed_segment = segment_data * window # Apply original window shape
            reconstructed_signal[start_idx_recon:end_idx_recon] += windowed_segment
            sum_of_windows[start_idx_recon:end_idx_recon] += normalized_window_for_sum # Use normalized for sum
            if not np.all(np.isfinite(reconstructed_signal[start_idx_recon:end_idx_recon])):
                reconstructed_signal[start_idx_recon:end_idx_recon] = np.nan_to_num(reconstructed_signal[start_idx_recon:end_idx_recon])
                logger.warning(f"Warn: NaN after adding chunk {i}")
        except Exception as add_e:
             logger.error(f"Error overlap-add chunk {i}: {add_e}")

        if i < len(chunks) - 1:
            current_recon_time_start += time_advance_per_chunk + tuning_delay

    logger.info("\nStitching loop complete.")
    return reconstructed_signal, sum_of_windows


def normalize_stitched_signal(raw_signal, sum_windows, target_rms):
    """
    Normalizes the raw stitched signal using sum_of_windows and applies final RMS scaling.

    Args:
        raw_signal (np.ndarray): The signal after overlap-add.
        sum_windows (np.ndarray): The sum of window functions.
        target_rms (float): The final desired RMS value.

    Returns:
        np.ndarray: The final normalized stitched signal. Returns None if input invalid.
    """
    logger.info("\n--- Normalizing reconstructed signal using Sum-of-Windows & Final RMS Scale ---")
    if raw_signal is None or sum_windows is None or len(raw_signal) != len(sum_windows) or len(raw_signal) == 0:
         logger.error("Error: Invalid input for stitched signal normalization.")
         return None

    # Stats before normalization
    finite_mask_raw = np.isfinite(raw_signal)
    rms_before_norm = np.nan
    max_abs_before_norm = np.nan
    if np.any(finite_mask_raw):
        rms_before_norm = np.sqrt(np.mean(np.abs(raw_signal[finite_mask_raw])**2))
        max_abs_before_norm = np.max(np.abs(raw_signal[finite_mask_raw]))

    logger.info("Signal Stats BEFORE Sum-of-Windows Normalization:")
    logger.info(f"  RMS (finite parts): {rms_before_norm:.4e}")
    logger.info(f"  Max Abs (finite parts): {max_abs_before_norm:.4e}")
    logger.info(f"  Sum of windows stats: Min={np.min(sum_windows):.4f}, Max={np.max(sum_windows):.4f}, Mean={np.mean(sum_windows):.4f}, Median={np.median(sum_windows):.4f}")

    # Sum-of-Windows normalization logic
    reliable_threshold = 1e-6
    reliable_indices = np.where(sum_windows >= reliable_threshold)[0]
    fallback_sum = 1.0
    if len(reliable_indices) > 0:
        median_sum = np.median(sum_windows[reliable_indices])
        if np.isfinite(median_sum) and median_sum > 1e-9: fallback_sum = median_sum
        else: fallback_sum = np.mean(sum_windows[reliable_indices]) # Use mean if median fails
        fallback_sum = fallback_sum if (np.isfinite(fallback_sum) and fallback_sum > 1e-9) else 1.0 # Final check
    logger.info(f"Using fallback divisor value: {fallback_sum:.4f} for unreliable regions.")

    sum_of_windows_divisor = sum_windows.copy()
    unreliable_mask = (sum_windows < reliable_threshold) | (~np.isfinite(sum_windows))
    sum_of_windows_divisor[unreliable_mask] = fallback_sum
    sum_of_windows_divisor[np.abs(sum_of_windows_divisor) < 1e-15] = fallback_sum

    normalized_signal = np.zeros_like(raw_signal)
    valid_divisor = np.isfinite(sum_of_windows_divisor) & (np.abs(sum_of_windows_divisor) > 1e-15)
    signal_to_divide = np.nan_to_num(raw_signal)
    np.divide(signal_to_divide, sum_of_windows_divisor, out=normalized_signal, where=valid_divisor)
    normalized_signal = np.nan_to_num(normalized_signal) # Final safety check for NaNs

    # Final RMS scaling
    logger.info("\n--- Applying Final RMS Scaling ---")
    rms_after_sow = np.sqrt(np.mean(np.abs(normalized_signal)**2))
    logger.info(f"RMS after Sum-of-Windows Norm: {rms_after_sow:.4e}")

    final_signal = normalized_signal.copy()
    if rms_after_sow > 1e-12:
        scale_factor = target_rms / rms_after_sow
        final_signal *= scale_factor
        logger.info(f"Applied final scaling factor {scale_factor:.4f} to match target RMS ({target_rms:.4e}).")
    else: logger.warning("Warning: RMS near zero after SoW norm. Skipping final RMS scaling.")

    # Final stats
    rms_final = np.sqrt(np.mean(np.abs(final_signal)**2))
    max_abs_final = np.max(np.abs(final_signal)) if len(final_signal)>0 else 0
    logger.info("\nSignal Stats AFTER Final Normalization:")
    logger.info(f"  Final RMS: {rms_final:.4e}")
    logger.info(f"  Final Max Abs: {max_abs_final:.4e}")

    logger.info("\nNormalization complete.")
    return final_signal