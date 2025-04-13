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
    """Normalizes each chunk individually to the target RMS before stitching. Uses logging."""
    logger.info("--- Normalizing chunks before stitching ---")
    if chunks is None:
        logger.error("Input 'chunks' is None.")
        return None
    if target_rms is None or target_rms <= 0:
         logger.error(f"Invalid target_rms: {target_rms}")
         return chunks # Return original potentially

    normalized_chunks = []
    logger.info("Chunk | RMS Before Norm | RMS After Norm")
    logger.info("------|-----------------|---------------")
    num_processed = 0
    for i, chunk in enumerate(chunks):
        rms_before = np.nan
        rms_after = np.nan
        norm_chunk = chunk # Default

        if not isinstance(chunk, np.ndarray) or chunk.ndim != 1:
            logger.warning(f"Chunk {i}: Invalid type ({type(chunk)}) or dimensions ({chunk.ndim}). Skipping.")
            normalized_chunks.append(chunk)
            continue

        if len(chunk) == 0:
             norm_chunk = chunk; rms_before = 0.0; rms_after = 0.0
             logger.info(f"{i:<5d} | --- EMPTY ---     | ---")
        elif not np.all(np.isfinite(chunk)):
             logger.warning(f"Chunk {i}: Contains non-finite values before norm. Zeroing.")
             norm_chunk = np.zeros_like(chunk); rms_after = 0.0
        else:
             try:
                 chunk_rms = np.sqrt(np.mean(np.abs(chunk)**2))
                 rms_before = chunk_rms
                 if chunk_rms > 1e-15:
                     scale = target_rms / chunk_rms
                     norm_chunk = chunk * scale
                     rms_after = np.sqrt(np.mean(np.abs(norm_chunk)**2))
                     logger.debug(f"Chunk {i}: RMS Before={rms_before:.4e}, Scale={scale:.4f}, RMS After={rms_after:.4e}")
                     if not np.isclose(rms_after, target_rms, rtol=1e-4):
                          logger.warning(f"Chunk {i}: RMS mismatch after norm ({rms_after:.4e} vs {target_rms:.4e}).")
                 else:
                     logger.info(f"Chunk {i}: RMS near zero ({rms_before:.4e}). Zeroing.")
                     norm_chunk = np.zeros_like(chunk); rms_after = 0.0
                 num_processed += 1
             except Exception as e:
                  logger.error(f"Error normalizing chunk {i}: {e}", exc_info=True)
                  norm_chunk = chunk # Fallback
                  rms_before = np.nan; rms_after = np.nan

        normalized_chunks.append(norm_chunk)
        logger.info(f"{i:<5d} | {rms_before:<15.4e} | {rms_after:.4e}")

    logger.info(f"--- Pre-Stitching Normalization Complete ({num_processed}/{len(chunks)} chunks processed) ---")
    return normalized_chunks


def stitch_signal(chunks, sdr_rate, recon_rate, overlap_factor, tuning_delay, window_type):
    """
    Stitches chunks using overlap-add method with specified window. Uses logging.
    Ensures calculations use the provided recon_rate.

    Args:
        chunks (list): List of normalized chunks (complex128 numpy arrays at recon_rate).
        sdr_rate (float): Original SDR sample rate (Hz). Used ONLY for chunk duration calculation fallback.
        recon_rate (float): Sample rate of the input chunks AND the target output rate (Hz).
        overlap_factor (float): The effective overlap factor used.
        tuning_delay (float): Tuning delay between chunks (seconds).
        window_type (str): Type of window for overlap-add (e.g., 'blackmanharris').

    Returns:
        tuple: (reconstructed_signal, sum_of_windows) or (None, None) on error.
    """
    logger.info("--- Performing Stitching (Overlap-Add) ---")
    # --- VVVVV Input Validation VVVVV ---
    if not chunks: logger.error("No chunks provided for stitching."); return None, None
    if any(not isinstance(c, np.ndarray) for c in chunks): logger.error("Input 'chunks' must be a list of numpy arrays."); return None, None
    if recon_rate is None or recon_rate <= 0: logger.error(f"Invalid reconstruction rate: {recon_rate}"); return None, None
    if sdr_rate is None or sdr_rate <= 0: logger.warning(f"Invalid sdr_rate: {sdr_rate}. Duration calc fallback may fail.")
    if overlap_factor < 0 or overlap_factor >= 1: logger.warning(f"Unusual overlap_factor: {overlap_factor:.3f}")
    if tuning_delay < 0: logger.warning(f"Negative tuning_delay: {tuning_delay*1e6:.1f} us")
    # --- ^^^^^ Input Validation ^^^^^ ---


    # --- Calculate Chunk Duration ---
    # Primarily use the length of the *first valid input chunk* and the *provided recon_rate*.
    chunk_duration_s = 0
    first_valid_chunk_idx = -1
    for idx, ch in enumerate(chunks):
        if ch is not None and len(ch) > 0:
            first_valid_chunk_idx = idx
            break

    if first_valid_chunk_idx != -1:
        chunk_duration_s = len(chunks[first_valid_chunk_idx]) / recon_rate # Use recon_rate here
        logger.info(f"Using chunk {first_valid_chunk_idx} length ({len(chunks[first_valid_chunk_idx])}) and recon_rate ({recon_rate/1e6:.1f}MHz) for duration estimate.")
        logger.debug(f"Estimated chunk duration: {chunk_duration_s*1e6:.3f} us")
    else:
        logger.error("No valid chunks found to estimate duration.")
        return None, None # Cannot proceed if no valid chunks

    # Handle single chunk case cleanly
    if len(chunks) == 1:
         logger.info("Only one chunk provided. Performing simple pass-through (no overlap/windowing).")
         single_chunk = np.asarray(chunks[0], dtype=complex)
         num_samples_recon = len(single_chunk)
         sum_of_windows = np.ones(num_samples_recon, dtype=float) # Effective window is rect=1
         logger.info("Stitching complete (single chunk).")
         return single_chunk, sum_of_windows
    # --- End Chunk Duration Calc ---


    # --- Calculate Reconstruction Buffer Size and Timing ---
    time_advance_per_chunk = chunk_duration_s * (1.0 - overlap_factor)
    if time_advance_per_chunk <= 0:
        logger.warning(f"Calculated time advance per chunk is zero or negative ({time_advance_per_chunk*1e6:.2f} us). Check overlap factor and duration.")
        # Might indicate overlap >= 1, allow proceeding but overlap logic might behave unexpectedly
    total_duration_recon = chunk_duration_s + max(0, len(chunks) - 1) * (time_advance_per_chunk + tuning_delay)
    # Use ceil to ensure buffer is large enough
    num_samples_recon = int(np.ceil(total_duration_recon * recon_rate))

    if num_samples_recon <= 0:
        logger.error(f"Invalid calculated reconstruction samples ({num_samples_recon}). Check rates/duration/delay.")
        return None, None

    logger.info(f"Reconstruction target buffer: {num_samples_recon} samples @ {recon_rate/1e6:.2f} MHz (Est. Duration: {total_duration_recon*1e6:.1f} us)")
    logger.info(f"Using stitching window: '{window_type}', Overlap: {overlap_factor*100:.1f}%, Tuning delay: {tuning_delay*1e6:.1f} us")
    logger.debug(f"Time advance per chunk (excl. delay): {time_advance_per_chunk*1e6:.3f} us")

    reconstructed_signal = np.zeros(num_samples_recon, dtype=complex)
    sum_of_windows = np.zeros(num_samples_recon, dtype=float)
    current_recon_time_start = 0.0
    # --- End Buffer Calc ---


    # --- Overlap-Add Loop ---
    for i, chunk_to_add in tqdm(enumerate(chunks), total=len(chunks), desc="Stitching"):
        if not isinstance(chunk_to_add, np.ndarray): logger.warning(f"Chunk {i} invalid type, skipping."); continue
        num_samples_in_chunk = len(chunk_to_add)
        if num_samples_in_chunk == 0: logger.debug(f"Chunk {i} is empty, skipping."); continue # Skip if chunk itself is empty

        start_idx_recon = int(round(current_recon_time_start * recon_rate))
        # Ensure indices are within bounds
        start_idx_recon = max(0, start_idx_recon)
        end_idx_recon = min(start_idx_recon + num_samples_in_chunk, num_samples_recon)
        actual_len = end_idx_recon - start_idx_recon # Number of samples from this chunk to use
        logger.debug(f"Chunk {i}: Start Idx={start_idx_recon}, End Idx={end_idx_recon}, Samples={actual_len}, ChunkLen={num_samples_in_chunk}")

        if actual_len <= 0:
            logger.debug(f"Chunk {i}: Skipping (zero length contribution to buffer).")
            # Still advance time if not the last chunk
            if i < len(chunks) - 1: current_recon_time_start += time_advance_per_chunk + tuning_delay
            continue

        # Get window
        window = np.ones(actual_len)
        if actual_len >= 2:
            try: window = sig.get_window(window_type, actual_len)
            except Exception as win_e: logger.warning(f"Failed get_window '{window_type}' Chk {i}: {win_e}. Using rect.")

        # Calculate normalized window for sum_of_windows
        window_rms = np.sqrt(np.mean(window**2))
        normalized_window_for_sum = window / (window_rms + 1e-15) # Avoid division by zero

        try:
            segment_data = chunk_to_add[:actual_len] # Take the correct number of samples
            if not np.all(np.isfinite(segment_data)):
                logger.warning(f"Chunk {i}: NaN/Inf in segment data BEFORE windowing. Applying nan_to_num.")
                segment_data = np.nan_to_num(segment_data)

            windowed_segment = segment_data * window
            logger.debug(f"Chunk {i}: Adding {actual_len} windowed samples to buffer range {start_idx_recon}:{end_idx_recon}")

            # Add to buffers
            reconstructed_signal[start_idx_recon:end_idx_recon] += windowed_segment
            sum_of_windows[start_idx_recon:end_idx_recon] += normalized_window_for_sum

            # Check for NaNs after addition
            if not np.all(np.isfinite(reconstructed_signal[start_idx_recon:end_idx_recon])):
                logger.warning(f"Chunk {i}: NaN/Inf detected AFTER adding windowed segment. Fixing range.")
                reconstructed_signal[start_idx_recon:end_idx_recon] = np.nan_to_num(reconstructed_signal[start_idx_recon:end_idx_recon])

        except ValueError as ve:
             logger.error(f"ValueError overlap-add Chk {i}: {ve}. Indices:{start_idx_recon}:{end_idx_recon}, WinLen:{len(window)}, SegLen:{len(segment_data)}", exc_info=True)
        except Exception as add_e:
             logger.error(f"Error overlap-add Chk {i}: {add_e}", exc_info=True)

        # Advance time for the *next* chunk's start
        if i < len(chunks) - 1:
            time_advance_this_step = time_advance_per_chunk + tuning_delay
            current_recon_time_start += time_advance_this_step
            logger.debug(f"Chunk {i}: Advancing time by {time_advance_this_step*1e6:.3f} us for next chunk.")

    logger.info("--- Stitching loop complete ---")
    return reconstructed_signal, sum_of_windows


def normalize_stitched_signal(raw_signal, sum_windows, target_rms):
    """
    Normalizes the raw stitched signal using sum_of_windows and applies final RMS scaling. Uses logging.
    """
    logger.info("--- Normalizing reconstructed signal using Sum-of-Windows & Final RMS Scale ---")
    if raw_signal is None or sum_windows is None or len(raw_signal) == 0 or len(raw_signal) != len(sum_windows):
         logger.error("Invalid input for stitched signal normalization.")
         return None

    logger.debug("Calculating stats before SoW normalization...")
    finite_mask_raw = np.isfinite(raw_signal)
    rms_before_norm, max_abs_before_norm = np.nan, np.nan
    if np.any(finite_mask_raw):
        valid_raw = raw_signal[finite_mask_raw]
        rms_before_norm = np.sqrt(np.mean(np.abs(valid_raw)**2))
        max_abs_before_norm = np.max(np.abs(valid_raw)) if len(valid_raw)>0 else 0.0
    else: logger.warning("Raw stitched signal contains no finite values.")

    logger.info("Signal Stats BEFORE Sum-of-Windows Normalization:")
    logger.info(f"  RMS (finite parts): {rms_before_norm:.4e}")
    logger.info(f"  Max Abs (finite parts): {max_abs_before_norm:.4e}")
    logger.info(f"  Sum of windows stats: Min={np.min(sum_windows):.4f}, Max={np.max(sum_windows):.4f}, Mean={np.mean(sum_windows):.4f}, Median={np.median(sum_windows):.4f}")

    # Sum-of-Windows normalization logic
    logger.debug("Calculating SoW divisor...")
    reliable_threshold = 1e-6
    reliable_indices = np.where(sum_windows >= reliable_threshold)[0]
    fallback_sum = 1.0
    if len(reliable_indices) > 0:
        median_sum = np.median(sum_windows[reliable_indices])
        if np.isfinite(median_sum) and median_sum > 1e-9: fallback_sum = median_sum
        else:
             mean_sum_reliable = np.mean(sum_windows[reliable_indices])
             if np.isfinite(mean_sum_reliable) and mean_sum_reliable > 1e-9: fallback_sum = mean_sum_reliable; logger.warning(f"Median sum unreliable. Using mean ({fallback_sum:.4f}).")
             else: logger.warning(f"Median/Mean unreliable. Using default fallback {fallback_sum}.")
    else: logger.warning(f"No reliable indices for sum_windows. Using default fallback {fallback_sum}.")
    logger.info(f"Using fallback divisor value: {fallback_sum:.4f} for unreliable regions.")

    sum_of_windows_divisor = sum_windows.copy()
    unreliable_mask = (sum_windows < reliable_threshold) | (~np.isfinite(sum_windows)) | (np.abs(sum_windows) < 1e-15)
    sum_of_windows_divisor[unreliable_mask] = fallback_sum
    # Double check for zeros after assigning fallback
    sum_of_windows_divisor[np.abs(sum_of_windows_divisor) < 1e-15] = fallback_sum

    logger.debug("Applying SoW division...")
    normalized_signal = np.zeros_like(raw_signal)
    signal_to_divide = np.nan_to_num(raw_signal)
    try:
        # Use np.divide for safe division
        valid_divisor_mask = np.abs(sum_of_windows_divisor) > 1e-15 # Where division is safe
        np.divide(signal_to_divide, sum_of_windows_divisor, out=normalized_signal, where=valid_divisor_mask)
    except Exception as div_e:
         logger.error(f"Error during SoW division: {div_e}", exc_info=True)
         return None # Cannot proceed
    # Final cleanup
    if not np.all(np.isfinite(normalized_signal)):
         logger.warning("Non-finite values detected AFTER SoW division. Applying nan_to_num.")
         normalized_signal = np.nan_to_num(normalized_signal)

    # Final RMS scaling
    logger.info("--- Applying Final RMS Scaling ---")
    rms_after_sow = np.sqrt(np.mean(np.abs(normalized_signal)**2)) if len(normalized_signal) > 0 else 0.0
    logger.info(f"RMS after Sum-of-Windows Norm: {rms_after_sow:.4e}")

    final_signal = normalized_signal.copy()
    if rms_after_sow > 1e-15: # Use smaller threshold
        scale_factor = target_rms / rms_after_sow
        final_signal *= scale_factor
        logger.info(f"Applied final scaling factor {scale_factor:.4f} to match target RMS ({target_rms:.4e}).")
    else: logger.warning("RMS near zero after SoW norm. Skipping final RMS scaling.")

    rms_final = np.sqrt(np.mean(np.abs(final_signal)**2)) if len(final_signal) > 0 else 0.0
    max_abs_final = 0.0
    if len(final_signal) > 0:
        finite_final = final_signal[np.isfinite(final_signal)]
        if len(finite_final) > 0: max_abs_final = np.max(np.abs(finite_final))

    logger.info("--- Signal Stats AFTER Final Normalization ---")
    logger.info(f"  Final RMS: {rms_final:.4e}")
    logger.info(f"  Final Max Abs: {max_abs_final:.4e}")
    logger.info("--- Normalization Complete ---")

    return final_signal