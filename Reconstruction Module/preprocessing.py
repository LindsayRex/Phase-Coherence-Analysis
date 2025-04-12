# preprocessing.py
"""Functions for initial signal preprocessing steps."""

import numpy as np
from scipy import signal as sig
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging # Import standard logging

# --- Get logger for this module ---
logger = logging.getLogger(__name__)
# --- End Get logger ---
# DO NOT call log_config.setup_logging() here

def scale_chunks(chunks, target_rms):
    """
    Scales each chunk to have a specific target RMS amplitude.

    Args:
        chunks (list): List of numpy arrays (complex128) containing IQ data.
        target_rms (float): The desired RMS value for each chunk.

    Returns:
        list: List of scaled numpy arrays (complex128). Returns original chunks if error.
    """
    logger.info("--- Correcting Initial Amplitude Scaling ---")
    scaled_chunks = []
    # Log the header using logger
    logger.info("Chunk | RMS Before | Scaling Factor | RMS After  | Max Abs (After)")
    logger.info("------|------------|----------------|------------|----------------")
    scaling_successful = True

    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, np.ndarray) or chunk.ndim != 1:
            logger.error(f"Chunk {i} is not a valid numpy array. Skipping scaling.")
            scaled_chunks.append(chunk) # Append original invalid chunk
            scaling_successful = False
            continue

        if len(chunk) == 0:
            scaled_chunks.append(chunk)
            logger.info(f"{i:<5d} | --- EMPTY ---    | ---            | --- EMPTY ---    | ---")
            continue

        if not np.all(np.isfinite(chunk)):
            logger.error(f"Chunk {i} contains non-finite values BEFORE scaling.")
            scaling_successful = False
            scaled_chunks.append(chunk)
            continue

        try: # Wrap calculation in try-except
            rms_before = np.sqrt(np.mean(np.abs(chunk)**2))
            max_abs_before = np.max(np.abs(chunk)) if len(chunk) > 0 else 0.0

            if rms_before < 1e-15: # Use a smaller threshold
                logger.info(f"{i:<5d} | {rms_before:.4e}       | SKIPPED (Zero) | {rms_before:.4e}       | {max_abs_before:.4e}")
                scaled_chunks.append(chunk)
                continue

            scaling_factor = target_rms / rms_before
            scaled_chunk = (chunk * scaling_factor).astype(np.complex128)

            if not np.all(np.isfinite(scaled_chunk)):
                logger.error(f"Chunk {i} non-finite values AFTER scaling (Factor: {scaling_factor:.4f}).")
                scaling_successful = False
                scaled_chunks.append(scaled_chunk) # Append erroneous scaled chunk
                continue

            rms_after = np.sqrt(np.mean(np.abs(scaled_chunk)**2))
            max_abs_after = np.max(np.abs(scaled_chunk)) if len(scaled_chunk) > 0 else 0.0

            # Check RMS closeness
            if not np.isclose(rms_after, target_rms, rtol=5e-3):
                logger.warning(f"Chunk {i} RMS scaling mismatch ({rms_after:.4e} vs {target_rms:.4e})")

            logger.info(f"{i:<5d} | {rms_before:.4e} | {scaling_factor:<14.4f} | {rms_after:.4e} | {max_abs_after:.4e}")
            scaled_chunks.append(scaled_chunk)

        except Exception as e:
            logger.error(f"Error scaling chunk {i}: {e}", exc_info=True)
            scaling_successful = False
            scaled_chunks.append(chunk) # Append original on error

    if not scaling_successful:
        logger.warning("Non-finite values or errors encountered during scaling. Results may be affected.")

    logger.info("--- Initial Amplitude Scaling Complete ---")
    return scaled_chunks


def upsample_chunks(chunks, sdr_rate, recon_rate, plot_first=False):
    """
    Upsamples chunks using polyphase filtering.

    Args:
        chunks (list): List of numpy arrays (complex128), typically after scaling.
        sdr_rate (float): Original sample rate of the chunks (Hz).
        recon_rate (float): Target sample rate for reconstruction (Hz).
        plot_first (bool): If True, plots the first upsampled chunk.

    Returns:
        list: List of upsampled numpy arrays (complex128).
    """
    logger.info("--- Upsampling chunks (using polyphase filter) ---")
    upsampled_chunks_list = []

    if sdr_rate is None or recon_rate is None or sdr_rate <= 0 or recon_rate <= 0:
        logger.error(f"Invalid sample rates for upsampling: SDR={sdr_rate}, Recon={recon_rate}. Returning zeros.")
        return [np.zeros_like(c, dtype=np.complex128) for c in chunks]

    # Calculate factors once if rates are constant
    try:
        # Use integer conversion robustly
        sdr_rate_int = int(np.round(sdr_rate))
        recon_rate_int = int(np.round(recon_rate))
        if sdr_rate_int <= 0 or recon_rate_int <= 0:
             raise ValueError("Rates must be positive for GCD.")
        common_divisor = np.gcd(recon_rate_int, sdr_rate_int)
        up_factor = recon_rate_int // common_divisor
        down_factor = sdr_rate_int // common_divisor
        logger.info(f"Upsampling Parameters: Up={up_factor}, Down={down_factor}")
    except Exception as e:
        logger.error(f"Could not calculate resampling factors: {e}. Aborting upsampling.", exc_info=True)
        return [np.zeros_like(c, dtype=np.complex128) for c in chunks]


    for i, chunk_data in tqdm(enumerate(chunks), total=len(chunks), desc="Upsampling"):
        if not isinstance(chunk_data, np.ndarray) or chunk_data.ndim != 1:
            logger.warning(f"Chunk {i} is not a valid numpy array. Skipping upsampling.")
            upsampled_chunks_list.append(np.array([], dtype=complex)) # Append empty
            continue

        if len(chunk_data) == 0:
             upsampled_chunks_list.append(chunk_data)
             logger.debug(f"Chunk {i}: Input is empty.")
             continue

        # Calculate target length based on duration and new rate
        # Use float rates for accurate duration/length calculation
        chunk_duration = len(chunk_data) / float(sdr_rate)
        num_samples_chunk_recon = int(round(chunk_duration * float(recon_rate)))
        logger.debug(f"Chunk {i}: Input Len={len(chunk_data)}, Target Len={num_samples_chunk_recon}")


        if len(chunk_data) < 2: # resample_poly needs at least 2 samples
            logger.warning(f"Chunk {i} too short ({len(chunk_data)} samples). Appending zeros.")
            upsampled_chunks_list.append(np.zeros(num_samples_chunk_recon, dtype=complex))
            continue

        try:
            rms_in = np.sqrt(np.mean(np.abs(chunk_data)**2))
            if rms_in < 1e-15:
                logger.info(f"Chunk {i}: Input RMS near zero. Appending zeros.")
                upsampled_chunks_list.append(np.zeros(num_samples_chunk_recon, dtype=complex))
                continue

            # --- Perform Upsampling ---
            logger.debug(f"Chunk {i}: Calling resample_poly...")
            # Ensure input is float64 for stability with filters
            resampled_real = sig.resample_poly(chunk_data.real.astype(np.float64), up=up_factor, down=down_factor, window=('kaiser', 5.0))
            resampled_imag = sig.resample_poly(chunk_data.imag.astype(np.float64), up=up_factor, down=down_factor, window=('kaiser', 5.0))
            upsampled_chunk = (resampled_real + 1j * resampled_imag).astype(np.complex128)
            logger.debug(f"Chunk {i}: Resample_poly output length: {len(upsampled_chunk)}")

            # --- Trim or pad to the exactly calculated target length ---
            current_len = len(upsampled_chunk)
            if current_len > num_samples_chunk_recon:
                logger.debug(f"Chunk {i}: Trimming from {current_len} to {num_samples_chunk_recon}")
                upsampled_chunk = upsampled_chunk[:num_samples_chunk_recon]
            elif current_len < num_samples_chunk_recon:
                pad_length = num_samples_chunk_recon - current_len
                logger.debug(f"Chunk {i}: Padding from {current_len} with {pad_length} zeros to {num_samples_chunk_recon}")
                upsampled_chunk = np.pad(upsampled_chunk, (0, pad_length), mode='constant')

            # --- Maintain RMS consistency ---
            current_rms = np.sqrt(np.mean(np.abs(upsampled_chunk)**2))
            if current_rms > 1e-15 and rms_in > 1e-15: # Avoid division by zero or scaling noise
                scale_correction = rms_in / current_rms
                upsampled_chunk *= scale_correction
                logger.debug(f"Chunk {i}: Applied RMS correction factor: {scale_correction:.4f}")
            elif rms_in > 1e-15: # Original had power, output doesn't - filter might have removed signal
                 logger.warning(f"Chunk {i}: Output RMS near zero after upsampling (Input RMS={rms_in:.4e}). Filter may have removed signal.")
                 upsampled_chunk.fill(0+0j) # Zero out if power disappeared

            upsampled_chunks_list.append(upsampled_chunk.copy())

            # --- Plotting first chunk ---
            if i == 0 and plot_first:
                logger.info("Plotting first upsampled chunk (polyphase method)...")
                try:
                    plt.figure(figsize=(12, 4))
                    time_axis_debug = np.arange(len(upsampled_chunk)) / recon_rate * 1e6
                    rms_out = np.sqrt(np.mean(np.abs(upsampled_chunk)**2))
                    max_abs_out = np.max(np.abs(upsampled_chunk)) if len(upsampled_chunk)>0 else 0
                    plt.plot(time_axis_debug, upsampled_chunk.real, label='Real')
                    plt.plot(time_axis_debug, upsampled_chunk.imag, label='Imag', alpha=0.7)
                    plt.title(f'First Upsampled Chunk (Polyphase, RMS={rms_out:.3e})')
                    plt.xlabel('Time (µs)'); plt.ylabel('Amp')
                    plt.legend(); plt.grid(True)
                    ylim_abs = max(max_abs_out * 1.2 if np.isfinite(max_abs_out) else 0.1, 0.05)
                    plt.ylim(-ylim_abs, ylim_abs)
                    plt.xlim(min(time_axis_debug, default=-0.1)-0.1, min(max(time_axis_debug, default=5.0), 5.0))
                    plt.show(block=False); plt.pause(0.1) # Consider saving instead if non-interactive
                except Exception as plot_e:
                    logger.error(f"Error plotting first upsampled chunk: {plot_e}", exc_info=False)


        except Exception as resample_e:
            logger.error(f"Error resampling chunk {i}: {resample_e}. Appending zeros.", exc_info=True)
            # Ensure target length exists even on error
            upsampled_chunks_list.append(np.zeros(num_samples_chunk_recon, dtype=complex))

    logger.info("--- Upsampling complete ---")
    return upsampled_chunks_list