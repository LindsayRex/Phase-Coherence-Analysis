# stitching.py
"""Functions for stitching chunks together using overlap-add."""

import numpy as np
from scipy import signal as sig
from tqdm import tqdm
import logging # Import standard logging

# --- Get logger for this module ---
logger = logging.getLogger(__name__)
# --- End Get logger ---
# DO NOT call log_config.setup_logging() here

def normalize_chunks_pre_stitch(chunks, target_rms):
    """Normalizes each chunk individually to the target RMS before stitching."""
    logger.info("--- Normalizing chunks before stitching ---")
    normalized_chunks = []
    logger.info("Chunk | RMS Before Norm | RMS After Norm")
    logger.info("------|-----------------|---------------")
    for i, chunk in enumerate(chunks):
        rms_before = np.nan
        rms_after = np.nan
        norm_chunk = chunk # Default to original if checks fail

        # Basic validation of input chunk
        if not isinstance(chunk, np.ndarray) or chunk.ndim != 1:
            logger.warning(f"Chunk {i}: Invalid input type ({type(chunk)}) or dimensions ({chunk.ndim}). Skipping normalization.")
            normalized_chunks.append(chunk) # Append original
            continue

        if len(chunk) == 0:
             norm_chunk = chunk; rms_before = 0.0; rms_after = 0.0
             logger.info(f"{i:<5d} | --- EMPTY ---     | ---") # Simplified log for empty
        elif not np.all(np.isfinite(chunk)):
             logger.warning(f"Chunk {i} contains non-finite values before norm. Zeroing.")
             norm_chunk = np.zeros_like(chunk); rms_after = 0.0
        else: # Process valid, non-empty, finite chunk
             try:
                 chunk_rms = np.sqrt(np.mean(np.abs(chunk)**2))
                 rms_before = chunk_rms
                 if chunk_rms > 1e-15: # Use smaller threshold
                     scale = target_rms / chunk_rms
                     norm_chunk = chunk * scale
                     # Verify RMS after scaling
                     rms_after = np.sqrt(np.mean(np.abs(norm_chunk)**2))
                     logger.debug(f"Chunk {i}: RMS Before={rms_before:.4e}, Scale={scale:.4f}, RMS After={rms_after:.4e}")
                     # Check closeness
                     if not np.isclose(rms_after, target_rms, rtol=1e-4):
                          logger.warning(f"Chunk {i}: RMS after norm ({rms_after:.4e}) not close to target ({target_rms:.4e}).")
                 else:
                     logger.info(f"Chunk {i}: RMS near zero ({rms_before:.4e}). Setting to zero.")
                     norm_chunk = np.zeros_like(chunk)
                     rms_after = 0.0
             except Exception as e:
                  logger.error(f"Error normalizing chunk {i}: {e}", exc_info=True)
                  norm_chunk = chunk # Fallback to original on error
                  rms_before = np.nan; rms_after = np.nan

        normalized_chunks.append(norm_chunk)
        # Log summary line (even if errors occurred, shows NaN)
        logger.info(f"{i:<5d} | {rms_before:<15.4e} | {rms_after:.4e}")

    logger.info("--- Pre-Stitching Normalization Complete ---")
    return normalized_chunks


def stitch_signal(chunks, sdr_rate, recon_rate, overlap_factor, tuning_delay, window_type):
    """
    Stitches chunks using overlap-add method with specified window. Uses logging.

    Args:
        chunks (list): List of normalized chunks (complex128 numpy arrays).
        sdr_rate (float): Original SDR sample rate (Hz). Used for duration/advance calc.
        recon_rate (float): Sample rate of the input chunks AND target output (Hz).
        overlap_factor (float): The effective overlap factor used.
        tuning_delay (float): Tuning delay between chunks (seconds).
        window_type (str): Type of window for overlap-add (e.g., 'blackmanharris').

    Returns:
        tuple: (reconstructed_signal, sum_of_windows) or (None, None) on error.
    """
    logger.info("--- Performing Stitching (Overlap-Add) ---")
    if not chunks:
        logger.error("No chunks provided for stitching.")
        return None, None
    if recon_rate is None or recon_rate <= 0:
         logger.error(f"Invalid reconstruction rate: {recon_rate}")
         return None, None

    # Calculate buffer size and timing
    chunk_duration_s = 0
    # Try getting duration from first chunk (assuming constant length/rate)
    if len(chunks[0]) > 0:
        chunk_duration_s = len(chunks[0]) / recon_rate
        logger.debug(f"Estimating chunk duration from first chunk: {chunk_duration_s*1e6:.1f} us")
    else:
         logger.warning("First chunk is empty, cannot estimate duration accurately.")
         # Fallback: Maybe use metadata if available? Requires passing metadata.
         # For now, assume a default or fail if needed later.

    if chunk_duration_s <= 0 and len(chunks) > 1: # Need duration if more than one chunk
         logger.error("Cannot determine chunk duration for stitching timing.")
         return None, None
    elif chunk_duration_s <= 0 and len(chunks) == 1:
         logger.info("Only one chunk provided, duration calculation skipped.")
         # Handle single chunk case - no overlap needed
         num_samples_recon = len(chunks[0])
         reconstructed_signal = np.zeros(num_samples_recon, dtype=complex)
         sum_of_windows = np.zeros(num_samples_recon, dtype=float)
         if num_samples_recon > 0:
             # Apply rectangular window of 1 for single chunk (effectively no windowing)
             reconstructed_signal = np.asarray(chunks[0], dtype=complex)
             sum_of_windows.fill(1.0)
         logger.info("Stitching complete (single chunk).")
         return reconstructed_signal, sum_of_windows

    # Calculate timing for multiple chunks
    time_advance_per_chunk = chunk_duration_s * (1.0 - overlap_factor)
    total_duration_recon = chunk_duration_s + max(0, len(chunks) - 1) * (time_advance_per_chunk + tuning_delay)
    num_samples_recon = int(round(total_duration_recon * recon_rate))

    if num_samples_recon <= 0:
        logger.error(f"Invalid calculated reconstruction samples ({num_samples_recon}). Check rates/duration.")
        return None, None

    logger.info(f"Reconstruction target buffer: {num_samples_recon} samples @ {recon_rate/1e6:.2f} MHz (Est. Duration: {total_duration_recon*1e6:.1f} us)")
    logger.info(f"Using stitching window: '{window_type}', Overlap factor: {overlap_factor:.3f}, Tuning delay: {tuning_delay*1e6:.1f} us")

    reconstructed_signal = np.zeros(num_samples_recon, dtype=complex)
    sum_of_windows = np.zeros(num_samples_recon, dtype=float)
    current_recon_time_start = 0.0

    for i, chunk_to_add in tqdm(enumerate(chunks), total=len(chunks), desc="Stitching"):
        start_idx_recon = int(round(current_recon_time_start * recon_rate))
        num_samples_in_chunk = len(chunk_to_add)
        end_idx_recon = min(start_idx_recon + num_samples_in_chunk, num_samples_recon)
        actual_len = end_idx_recon - start_idx_recon
        logger.debug(f"Chunk {i}: Start Idx={start_idx_recon}, End Idx={end_idx_recon}, Samples={actual_len}")


        if actual_len <= 0 or num_samples_in_chunk == 0:
            logger.debug(f"Chunk {i}: Skipping (zero length contribution or empty chunk).")
            if i < len(chunks) - 1: current_recon_time_start += time_advance_per_chunk + tuning_delay
            continue

        # Get window
        window = np.ones(actual_len) # Default rect window
        if actual_len >= 2:
            try: window = sig.get_window(window_type, actual_len)
            except Exception as win_e: logger.warning(f"Failed get_window '{window_type}' Chk {i}: {win_e}. Using rect.")

        # Calculate normalized window for sum (RMS=1)
        window_rms = np.sqrt(np.mean(window**2))
        normalized_window_for_sum = np.ones_like(window)
        if window_rms > 1e-12: normalized_window_for_sum = window / window_rms
        else: logger.debug(f"Chunk {i}: Window RMS is near zero.")

        try:
            segment_data = chunk_to_add[:actual_len]
            if not np.all(np.isfinite(segment_data)):
                logger.warning(f"Chunk {i}: NaN/Inf detected in segment data BEFORE windowing. Applying nan_to_num.")
                segment_data = np.nan_to_num(segment_data)

            windowed_segment = segment_data * window
            logger.debug(f"Chunk {i}: Adding {actual_len} samples to buffer range {start_idx_recon}:{end_idx_recon}")

            reconstructed_signal[start_idx_recon:end_idx_recon] += windowed_segment
            sum_of_windows[start_idx_recon:end_idx_recon] += normalized_window_for_sum

            # Check for NaNs introduced by addition (shouldn't happen if inputs cleaned)
            if not np.all(np.isfinite(reconstructed_signal[start_idx_recon:end_idx_recon])):
                logger.warning(f"Chunk {i}: NaN/Inf detected AFTER adding windowed segment. Applying nan_to_num to affected range.")
                reconstructed_signal[start_idx_recon:end_idx_recon] = np.nan_to_num(reconstructed_signal[start_idx_recon:end_idx_recon])

        except ValueError as ve:
             logger.error(f"ValueError during overlap-add chunk {i}: {ve}. Indices: {start_idx_recon}:{end_idx_recon}, Len: {actual_len}", exc_info=True)
        except Exception as add_e:
             logger.error(f"Error during overlap-add chunk {i}: {add_e}", exc_info=True)

        # Advance time for the *next* chunk's start
        if i < len(chunks) - 1:
            time_advance_this_step = time_advance_per_chunk + tuning_delay
            current_recon_time_start += time_advance_this_step
            logger.debug(f"Chunk {i}: Advancing time by {time_advance_this_step*1e6:.2f} us for next chunk.")

    logger.info("Stitching loop complete.")
    return reconstructed_signal, sum_of_windows


def normalize_stitched_signal(raw_signal, sum_windows, target_rms):
    """
    Normalizes the raw stitched signal using sum_of_windows and applies final RMS scaling. Uses logging.

    Args:
        raw_signal (np.ndarray): The signal after overlap-add.
        sum_windows (np.ndarray): The sum of window functions.
        target_rms (float): The final desired RMS value.

    Returns:
        np.ndarray: The final normalized stitched signal. Returns None if input invalid.
    """
    logger.info("--- Normalizing reconstructed signal using Sum-of-Windows & Final RMS Scale ---")
    if raw_signal is None or sum_windows is None or len(raw_signal) == 0 or len(raw_signal) != len(sum_windows):
         logger.error("Invalid input for stitched signal normalization.")
         return None

    finite_mask_raw = np.isfinite(raw_signal)
    rms_before_norm, max_abs_before_norm = np.nan, np.nan
    if np.any(finite_mask_raw):
        rms_before_norm = np.sqrt(np.mean(np.abs(raw_signal[finite_mask_raw])**2))
        max_abs_before_norm = np.max(np.abs(raw_signal[finite_mask_raw]))
    else: logger.warning("Raw stitched signal contains no finite values before normalization.")

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
        else:
             mean_sum_reliable = np.mean(sum_windows[reliable_indices])
             if np.isfinite(mean_sum_reliable) and mean_sum_reliable > 1e-9:
                  fallback_sum = mean_sum_reliable
                  logger.warning(f"Median sum ({median_sum}) not reliable. Using mean ({fallback_sum:.4f}) as fallback.")
             else: logger.warning(f"Median and Mean of reliable sum_of_windows not reliable. Using default fallback {fallback_sum}.")
    else: logger.warning(f"No reliable indices found for sum_of_windows. Using default fallback {fallback_sum}.")
    logger.info(f"Using fallback divisor value: {fallback_sum:.4f} for unreliable regions.")

    sum_of_windows_divisor = sum_windows.copy()
    unreliable_mask = (sum_windows < reliable_threshold) | (~np.isfinite(sum_windows))
    sum_of_windows_divisor[unreliable_mask] = fallback_sum
    sum_of_windows_divisor[np.abs(sum_of_windows_divisor) < 1e-15] = fallback_sum

    normalized_signal = np.zeros_like(raw_signal)
    valid_divisor = np.isfinite(sum_of_windows_divisor) & (np.abs(sum_of_windows_divisor) > 1e-15)
    signal_to_divide = np.nan_to_num(raw_signal) # Replace NaNs/Infs with 0 before dividing
    np.divide(signal_to_divide, sum_of_windows_divisor, out=normalized_signal, where=valid_divisor)
    if not np.all(np.isfinite(normalized_signal)):
         logger.warning("Non-finite values detected AFTER robust division. Applying nan_to_num.")
         normalized_signal = np.nan_to_num(normalized_signal)

    # Final RMS scaling
    logger.info("--- Applying Final RMS Scaling ---")
    rms_after_sow = np.sqrt(np.mean(np.abs(normalized_signal)**2)) if len(normalized_signal) > 0 else 0.0
    logger.info(f"RMS after Sum-of-Windows Norm: {rms_after_sow:.4e}")

    final_signal = normalized_signal.copy()
    if rms_after_sow > 1e-12:
        scale_factor = target_rms / rms_after_sow
        final_signal *= scale_factor
        logger.info(f"Applied final scaling factor {scale_factor:.4f} to match target RMS ({target_rms:.4e}).")
    else: logger.warning("RMS near zero after SoW norm. Skipping final RMS scaling.")

    rms_final = np.sqrt(np.mean(np.abs(final_signal)**2)) if len(final_signal) > 0 else 0.0
    max_abs_final = np.max(np.abs(final_signal)) if len(final_signal) > 0 and np.any(np.isfinite(final_signal)) else 0.0
    logger.info("--- Signal Stats AFTER Final Normalization ---")
    logger.info(f"  Final RMS: {rms_final:.4e}")
    logger.info(f"  Final Max Abs: {max_abs_final:.4e}")
    logger.info("--- Normalization Complete ---")

    return final_signal