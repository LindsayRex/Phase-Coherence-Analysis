# spectral_stitching.py
"""
Performs frequency domain stitching of overlapping signal chunks.
"""

import numpy as np
import logging
from tqdm import tqdm
import sys # Added for exit potentially

# --- CuPy Import and Check ---
cupy_available = False
try:
    import cupy as cp
    try:
        cp.cuda.Device(0).use()
        cupy_available = True
    except Exception: cupy_available = False
except ImportError: cupy_available = False
# --- End CuPy Import ---

logger = logging.getLogger(__name__)

def get_ideal_fft_size(min_size):
    """ Calculates the next power of 2 for efficient FFT. """
    if min_size <= 0: return 256
    return int(2**np.ceil(np.log2(min_size)))

# --- VVVVV ADDED Blackman-Harris window function (if needed) VVVVV ---
def blackmanharris(N, xp=np):
    """ Generate Blackman-Harris window coefficients using NumPy or CuPy """
    logger.debug(f"Generating Blackman-Harris window (N={N}) using {xp.__name__}")
    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168
    n = xp.arange(N)
    # Ensure denominator is not zero for N=1 case
    denom = N - 1 if N > 1 else 1
    cos2 = xp.cos(2 * np.pi * n / denom)
    cos4 = xp.cos(4 * np.pi * n / denom)
    cos6 = xp.cos(6 * np.pi * n / denom)
    return a0 - a1 * cos2 + a2 * cos4 - a3 * cos6
# --- ^^^^^ ADDED Blackman-Harris window function ^^^^^ ---

def frequency_domain_stitch(
    sdr_rate_chunks,
    metadata,
    global_attrs,
    fs_recon_final,
    freq_domain_window='hann', # Default to Hann
    target_rms=None
    ):
    """
    Stitches phase-corrected SDR rate chunks in the frequency domain with windowing.
    """
    logger.info(f"--- Starting Frequency Domain Stitching (Output Rate: {fs_recon_final/1e6:.1f} MHz) ---")

    # Validation & Parameter Extraction (as before)
    if not sdr_rate_chunks or not metadata or len(sdr_rate_chunks) != len(metadata): logger.error("Invalid inputs."); return None
    num_chunks = len(sdr_rate_chunks); sdr_rate = global_attrs.get('sdr_sample_rate_hz'); f_rf_center = global_attrs.get('rf_center_freq_hz')
    tuning_delay = global_attrs.get('tuning_delay_s', 0.0); overlap_factor = global_attrs.get('overlap_factor', 0.1)
    if sdr_rate is None or fs_recon_final is None or f_rf_center is None or sdr_rate <= 0 or fs_recon_final <= 0: logger.error("Missing/Invalid rates/center freq."); return None
    logger.info(f"Input rate: {sdr_rate/1e6:.1f} MHz. Output rate: {fs_recon_final/1e6:.1f} MHz. Window: '{freq_domain_window}'")
    xp = cp if cupy_available else np; logger.info(f"Using {'GPU (CuPy)' if cupy_available else 'CPU (NumPy)'}.")

    # Calculate Final Buffer Size & Duration (as before)
    chunk_len_sdr = len(sdr_rate_chunks[0]) if len(sdr_rate_chunks) > 0 else 0
    if chunk_len_sdr == 0: logger.error("First chunk is empty."); return None
    chunk_duration_s = chunk_len_sdr / sdr_rate; time_advance_per_chunk = chunk_duration_s * (1.0 - overlap_factor)
    total_duration_recon = chunk_duration_s + max(0, num_chunks - 1) * (time_advance_per_chunk + tuning_delay)
    num_samples_final = int(round(total_duration_recon * fs_recon_final)); N_fft_final = get_ideal_fft_size(num_samples_final)
    logger.info(f"Target output samples: {num_samples_final}, Final FFT size: {N_fft_final}")
    if N_fft_final <= 0: logger.error("Invalid final FFT size."); return None

    # Initialize Final Spectrum Buffers
    try:
        final_spectrum = xp.zeros(N_fft_final, dtype=xp.complex128)
        final_window_sum = xp.zeros(N_fft_final, dtype=xp.float64)
        logger.debug(f"Initialized final spectrum buffers (size {N_fft_final}) on {'GPU' if cupy_available else 'CPU'}.")
    except Exception as e: logger.error(f"Failed to allocate spectrum buffers: {e}", exc_info=True); return None

    # --- Process Each Chunk ---
    for i in tqdm(range(num_chunks), desc="Spectral Stitching"):
        chunk_cpu = np.asarray(sdr_rate_chunks[i]); meta = metadata[i]
        chunk_rf_center = meta.get('rf_center_freq_hz');
        if chunk_rf_center is None: logger.warning(f"Chk {i}: Missing RF center. Skipping."); continue
        if len(chunk_cpu) < 2: logger.warning(f"Chk {i}: Too short. Skipping."); continue
        logger.debug(f"Processing Chk {i}: RF={chunk_rf_center/1e9:.4f} GHz, Len={len(chunk_cpu)}")

        # FFT of the SDR Rate Chunk
        n_fft_chunk = get_ideal_fft_size(len(chunk_cpu))
        logger.debug(f"  Chk {i}: FFT (N={n_fft_chunk})")
        try:
            chunk_dev = xp.asarray(chunk_cpu); fft_chunk = xp.fft.fft(chunk_dev, n=n_fft_chunk); del chunk_dev
            freqs_chunk_baseband = xp.fft.fftfreq(n_fft_chunk, d=1/sdr_rate)
        except Exception as e: logger.error(f"FFT error Chk {i}: {e}"); continue

        # Calculate Target Bin Indices (as before)
        freq_shift = chunk_rf_center - f_rf_center
        chunk_freqs_in_final_axis = freqs_chunk_baseband + freq_shift
        indices_final = xp.round((chunk_freqs_in_final_axis / fs_recon_final) * N_fft_final).astype(int) % N_fft_final
        unique_indices, unique_map = xp.unique(indices_final, return_index=True)
        logger.debug(f"  Chk {i}: Mapped {len(unique_map)} points.")

        # --- VVVVV Apply Frequency Domain Window VVVVV ---
        logger.debug(f"  Chk {i}: Applying '{freq_domain_window}' window (FFT domain)...")
        window_fft = None
        if freq_domain_window is None or freq_domain_window.lower() == 'rect':
            window_fft = xp.ones(n_fft_chunk, dtype=xp.float64)
        elif freq_domain_window.lower() == 'hann':
            # Need to apply Hann in time domain equivalent or create freq version
            # Standard Hann window is time domain, need its FFT or apply carefully
            # Easier: apply window *before* FFT (equivalent to convolution in freq domain)
            # Let's modify to apply window *before* FFT for simplicity & correctness
            logger.warning("Applying Hann window *before* FFT (time domain) for simplicity.")
            # Recalculate FFT with time domain window:
            try:
                 chunk_dev = xp.asarray(chunk_cpu)
                 time_window = xp.hanning(len(chunk_cpu)) # Hann window for original length
                 fft_chunk = xp.fft.fft(chunk_dev * time_window, n=n_fft_chunk) # FFT of time-windowed chunk
                 del chunk_dev
                 # For frequency domain combining, the effective window applied to the spectrum
                 # is related to the FFT of the time-domain window. For overlap-add,
                 # we need windows that sum to 1 in the overlap. Let's use sqrt-Hann for overlap-add.
                 # Or stick to simple averaging via window_sum normalization.
                 # For simple averaging, use the window shape itself for the sum.
                 # Let's create the frequency domain window for summing based on time window FFT
                 window_fft_time = xp.hanning(n_fft_chunk) # Create window of FFT length for summing
                 window_fft_for_sum = window_fft_time # Use this shape for summing weights

                 # No separate spectral multiplication needed if windowed in time
                 fft_chunk_windowed = fft_chunk

            except Exception as e:
                 logger.error(f"Error re-computing FFT with window for chunk {i}: {e}")
                 continue
        elif freq_domain_window.lower() == 'blackmanharris':
              logger.warning("Applying Blackman-Harris window *before* FFT (time domain).")
              try:
                 chunk_dev = xp.asarray(chunk_cpu)
                 time_window = blackmanharris(len(chunk_cpu), xp=xp) # Use custom function
                 fft_chunk = xp.fft.fft(chunk_dev * time_window, n=n_fft_chunk)
                 del chunk_dev
                 window_fft_time = blackmanharris(n_fft_chunk, xp=xp) # Window shape for sum
                 window_fft_for_sum = window_fft_time
                 fft_chunk_windowed = fft_chunk
              except Exception as e:
                 logger.error(f"Error re-computing FFT with window for chunk {i}: {e}")
                 continue
        else:
             logger.warning(f"Unsupported freq domain window '{freq_domain_window}'. Using time-domain Rectangular (no window before FFT).")
             # Recalculate FFT without window if not done before
             if 'fft_chunk' not in locals(): # Check if FFT needs recalculating
                  try:
                       chunk_dev = xp.asarray(chunk_cpu); fft_chunk = xp.fft.fft(chunk_dev, n=n_fft_chunk); del chunk_dev
                  except Exception as e: logger.error(f"FFT error Chk {i}: {e}"); continue
             window_fft_for_sum = xp.ones(n_fft_chunk, dtype=xp.float64) # Use Rect window for sum
             fft_chunk_windowed = fft_chunk
        # --- ^^^^^ Apply Frequency Domain Window ^^^^^ ---


        # --- Add to Final Spectrum ---
        try:
            # Add windowed spectrum to final buffer
            final_spectrum[unique_indices] += fft_chunk_windowed[unique_map]
            # Add the corresponding window shape used for averaging
            final_window_sum[unique_indices] += window_fft_for_sum[unique_map]
            logger.debug(f"  Chk {i}: Added spectrum to final buffer.")
        except IndexError: logger.error(f"Chk {i}: Indexing error during spectral addition."); continue
        except Exception as e: logger.error(f"Error adding Chk {i} spectrum: {e}"); continue

        # --- Clean up ---
        del fft_chunk, freqs_chunk_baseband, chunk_freqs_in_final_axis, indices_final, unique_indices, unique_map, fft_chunk_windowed, window_fft_for_sum
        if cupy_available: cp.get_default_memory_pool().free_all_blocks()
        # --- End Chunk Loop ---

    logger.info("Spectral assembly complete.")

    # --- VVVVV MODIFIED Normalization Logic VVVVV ---
    logger.info("Normalizing final spectrum by window sum...")
    epsilon = 1e-15

    # Create output array (copying is safest)
    normalized_final_spectrum = final_spectrum.copy()  # Work on a copy

    # Identify where the divisor is valid
    valid_divisor_mask = final_window_sum > epsilon

    num_unnormalized = N_fft_final - xp.sum(valid_divisor_mask)
    if num_unnormalized > 0:
        logger.warning(f"{num_unnormalized}/{N_fft_final} final spectrum bins have near-zero window sum. Setting output to zero.")

    # Perform division only on valid elements
    normalized_final_spectrum[valid_divisor_mask] = final_spectrum[valid_divisor_mask] / final_window_sum[valid_divisor_mask]

    # Explicitly set invalid elements to zero
    normalized_final_spectrum[~valid_divisor_mask] = 0.0

    logger.debug("Final spectrum normalized.")

    # We no longer need the original final_spectrum or the sum array
    del final_spectrum, final_window_sum
    if cupy_available: cp.get_default_memory_pool().free_all_blocks()  # Clear intermediates
    # --- ^^^^^ MODIFIED Normalization Logic ^^^^^ ---

    # --- Inverse FFT to get Time Domain Signal ---
    logger.info(f"Computing Inverse FFT (N={N_fft_final})...")
    try:
        # Use the normalized spectrum for IFFT
        final_stitched_signal_full = xp.fft.ifft(normalized_final_spectrum)
        logger.debug("IFFT complete.")
        del normalized_final_spectrum  # Free memory

        # Trim to expected number of samples
        if len(final_stitched_signal_full) > num_samples_final:
            final_stitched_signal = final_stitched_signal_full[:num_samples_final]
            del final_stitched_signal_full
        elif len(final_stitched_signal_full) < num_samples_final:
            logger.warning(f"IFFT length ({len(final_stitched_signal_full)}) < expected ({num_samples_final}). Padding.")
            pad_len = num_samples_final - len(final_stitched_signal_full)
            final_stitched_signal = xp.pad(final_stitched_signal_full, (0, pad_len))
            del final_stitched_signal_full
        else:
            final_stitched_signal = final_stitched_signal_full
        
        logger.info(f"IFFT complete. Final time signal length: {len(final_stitched_signal)}")
    except Exception as e:
        logger.error(f"Error during inverse FFT or trimming: {e}", exc_info=True)
        # Clean up potentially partially created arrays
        if 'normalized_final_spectrum' in locals():
            del normalized_final_spectrum
        if 'final_stitched_signal_full' in locals():
            del final_stitched_signal_full
        if cupy_available:
            cp.get_default_memory_pool().free_all_blocks()
        return None

    # --- Final RMS Scaling ---
    if target_rms is not None and target_rms > 0:
        logger.info("Applying final RMS scaling...")
        # Compute RMS on device
        current_rms = xp.sqrt(xp.mean(xp.abs(final_stitched_signal)**2))
        logger.debug(f"RMS before final scaling: {float(current_rms.get()) if cupy_available else current_rms:.4e}")
        if current_rms > 1e-15:
            scale_factor = target_rms / current_rms
            final_stitched_signal *= scale_factor
            logger.info(f"Applied final RMS scale factor: {scale_factor:.4f}")
        else:
            logger.warning("Final signal RMS near zero, skipping scaling.")

    # --- Transfer Result to CPU ---
    if cupy_available:
        logger.debug("Transferring final signal GPU->CPU...")
        result_cpu = cp.asnumpy(final_stitched_signal)
        del final_stitched_signal
        cp.get_default_memory_pool().free_all_blocks()
        return result_cpu
    else:
        return final_stitched_signal  # Already NumPy