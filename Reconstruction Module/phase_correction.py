# phase_correction.py
"""Functions for correcting phase discontinuities between chunks using GPU."""

import numpy as np
from tqdm import tqdm
import pywt
import sys
import logging # Import standard logging

# --- Get logger for this module ---
logger = logging.getLogger(__name__)
# --- End Get logger ---
# DO NOT call log_config.setup_logging() here

# --- CuPy Import and Check ---
cupy_available = False # Default to False
try:
    import cupy as cp
    try:
        cp.cuda.Device(0).use()
        cupy_available = True
        logger.info("CuPy found and GPU available. Using GPU acceleration for correlation.")
    except cp.cuda.runtime.CUDARuntimeError as e:
        logger.warning(f"CuPy found, but GPU unavailable/error: {e}. Correlation cannot use GPU.")
    except Exception as e_init:
         logger.warning(f"CuPy found, but error during GPU init: {e_init}. Correlation cannot use GPU.")
except ImportError:
    logger.warning("CuPy not found. Correlation will run on CPU using NumPy.")
# --- End CuPy Import ---


def correct_phase_correlation(chunks, sdr_rate, recon_rate, base_overlap_factor, min_overlap_factor, max_lag_fraction, max_lag_abs):
    """
    Corrects chunk phases using GPU-accelerated time-domain cross-correlation
    with lag estimation and phasor averaging for phase estimation.

    Args:
        chunks (list): List of upsampled chunk data (complex128 numpy arrays).
        sdr_rate (float): Original SDR sample rate (Hz).
        recon_rate (float): Sample rate of the input chunks (Hz).
        base_overlap_factor (float): Original overlap factor from metadata.
        min_overlap_factor (float): Minimum overlap factor to use for correlation.
        max_lag_fraction (float): Max lag as fraction of overlap samples.
        max_lag_abs (int): Absolute max lag samples.

    Returns:
        tuple: (corrected_chunks, estimated_cumulative_phases)
               - corrected_chunks (list): List of numpy arrays (CPU) with phase correction applied.
               - estimated_cumulative_phases (list): Estimated cumulative phase offset for each chunk (CPU floats).
               Returns original chunks if GPU correlation cannot proceed.
    """
    logger.info("--- Performing Improved Phase Correlation (Correlation/Phasor Avg) ---") # Renamed slightly

    # Decide backend based on CuPy availability
    if cupy_available:
        logger.info("Using GPU (CuPy) for correlation loop.")
        xp = cp
    else:
        logger.warning("Using CPU (NumPy) for correlation loop. This might be slow.")
        xp = np

    if not chunks:
        logger.error("No chunks provided for phase correlation.")
        return [], []

    # Calculate overlap samples (on CPU)
    chunk_duration_s = 0
    if len(chunks) > 0 and recon_rate > 0:
         chunk_duration_s = len(chunks[0]) / recon_rate
    elif len(chunks) > 0 and sdr_rate > 0: # Fallback estimate
         logger.warning("Estimating chunk duration from SDR rate (less accurate).")
         chunk_duration_s = len(chunks[0]) / sdr_rate
    else:
         logger.warning("Cannot determine chunk duration for overlap calc.")

    effective_overlap = max(min_overlap_factor, base_overlap_factor)
    overlap_samples_recon = int(round(chunk_duration_s * effective_overlap * recon_rate)) if recon_rate is not None else 0
    logger.info(f"Overlap samples for Correlation: {overlap_samples_recon}")

    # Ensure first chunk is numpy array on CPU
    corrected_chunks = [np.asarray(chunks[0])]
    estimated_cumulative_phases = [0.0]

    # Define GPU variables list for cleanup (only relevant if cupy_available)
    gpu_vars_to_clean = ['prev_segment_gpu', 'curr_segment_gpu', 'xcorr_mag_gpu',
                         'xcorr_slice_gpu', 'best_lag_idx_in_search_gpu',
                         'aligned_curr_gpu', 'aligned_prev_gpu',
                         'complex_prod_gpu', 'phasors_gpu', 'avg_phasor_gpu',
                         'delta_phi_gpu']

    for i in tqdm(range(1, len(chunks)), desc="Phase Correlation"):
        # Get previous corrected chunk (CPU) and current original chunk (CPU)
        prev_chunk_cpu = np.asarray(corrected_chunks[i-1]) # Previous is already corrected
        curr_chunk_cpu = np.asarray(chunks[i]) # Current is original input

        # Check overlap validity (CPU)
        if overlap_samples_recon <= 1 or len(prev_chunk_cpu) < overlap_samples_recon or len(curr_chunk_cpu) < overlap_samples_recon:
            status = f"samples={overlap_samples_recon}" if overlap_samples_recon <= 1 else f"len prev={len(prev_chunk_cpu)},curr={len(curr_chunk_cpu)}"
            logger.warning(f"Insufficient overlap ({status}) for chunk {i}. Skipping correction step.")
            estimated_cumulative_phases.append(estimated_cumulative_phases[-1])
            corrected_chunks.append(curr_chunk_cpu) # Append the uncorrected chunk
            continue

        # Extract overlapping segments (CPU)
        prev_segment_cpu = prev_chunk_cpu[-overlap_samples_recon:]
        curr_segment_cpu = curr_chunk_cpu[:overlap_samples_recon]

        # Initialize defaults
        delta_phi = 0.0
        best_lag = 0

        # --- GPU/CPU Processing Block ---
        # Initialize device variables (set to None if using CPU)
        prev_segment_dev, curr_segment_dev = None, None
        aligned_curr_dev, aligned_prev_dev = None, None
        # ... other device vars ...

        try:
            # Transfer data to device (GPU or CPU)
            prev_segment_dev = xp.asarray(prev_segment_cpu)
            curr_segment_dev = xp.asarray(curr_segment_cpu)

            # Compute cross-correlation of magnitudes on device
            max_lag = min(max_lag_abs, int(overlap_samples_recon * max_lag_fraction))
            xcorr_mag_dev = xp.correlate(xp.abs(curr_segment_dev), xp.abs(prev_segment_dev), mode='full')

            # Lag calculation (indices on CPU, argmax on device)
            n_curr = len(curr_segment_dev); n_prev = len(prev_segment_dev)
            lags = np.arange(-(n_curr - 1), n_prev) # Always CPU
            search_indices = np.where(np.abs(lags) <= max_lag)[0]

            if len(search_indices) > 0:
                valid_search_indices = search_indices[(search_indices >= 0) & (search_indices < len(xcorr_mag_dev))]
                if len(valid_search_indices) > 0:
                    xcorr_slice_dev = xcorr_mag_dev[valid_search_indices]
                    best_lag_idx_in_search_dev = xp.argmax(xp.abs(xcorr_slice_dev))
                    best_lag_idx_in_search = int(best_lag_idx_in_search_dev.get()) if cupy_available else int(best_lag_idx_in_search_dev)
                    best_overall_idx = valid_search_indices[best_lag_idx_in_search]
                    best_lag = lags[best_overall_idx] # CPU lag

            # Align segments on device
            if best_lag > 0:
                aligned_curr_dev = curr_segment_dev[best_lag:]
                aligned_prev_dev = prev_segment_dev[:len(aligned_curr_dev)]
            elif best_lag < 0:
                aligned_prev_dev = prev_segment_dev[abs(best_lag):]
                aligned_curr_dev = curr_segment_dev[:len(aligned_prev_dev)]
            else:
                aligned_curr_dev = curr_segment_dev
                aligned_prev_dev = prev_segment_dev

            # Estimate phase using Phasor Averaging on device
            if len(aligned_curr_dev) > 0 and len(aligned_prev_dev) > 0:
                min_len = min(len(aligned_curr_dev), len(aligned_prev_dev))
                complex_prod_dev = aligned_curr_dev[:min_len] * xp.conj(aligned_prev_dev[:min_len])
                magnitudes_dev = xp.abs(complex_prod_dev) + 1e-20
                phasors_dev = complex_prod_dev / magnitudes_dev
                avg_phasor_dev = xp.mean(phasors_dev)
                if xp.abs(avg_phasor_dev) > 1e-12:
                    delta_phi_dev = xp.angle(avg_phasor_dev)
                    delta_phi = float(delta_phi_dev.get()) if cupy_available else float(delta_phi_dev) # CPU float

        except Exception as corr_e:
             logger.error(f"ERROR during correlation computation for chunk {i}: {corr_e}. Using delta_phi=0.", exc_info=True)
             delta_phi = 0.0
             best_lag = 0
        finally:
             # --- Clear GPU memory (only if CuPy was used) ---
             if cupy_available:
                 # Define list of GPU vars created in the try block
                 gpu_vars_loop = ['prev_segment_dev', 'curr_segment_dev', 'xcorr_mag_dev',
                                  'xcorr_slice_dev', 'best_lag_idx_in_search_dev',
                                  'aligned_curr_dev', 'aligned_prev_dev',
                                  'complex_prod_dev', 'magnitudes_dev', 'phasors_dev',
                                  'avg_phasor_dev', 'delta_phi_dev']
                 for var_name in gpu_vars_loop:
                      if var_name in locals() and isinstance(locals()[var_name], cp.ndarray):
                          try:
                              del locals()[var_name]
                          except NameError: pass # Variable might not exist if error occurred early
                 try:
                      mempool = cp.get_default_memory_pool()
                      mempool.free_all_blocks()
                 except Exception as mem_e:
                      logger.warning(f"Error freeing CuPy memory pool: {mem_e}")
        # --- End GPU/CPU Processing Block ---

        # Diagnostic Log
        logger.info(f"  Chunk {i}: Est. Lag = {best_lag}, Delta Phi (Phasor Avg, deg) = {np.rad2deg(delta_phi):.4f}")

        # Update cumulative phase (CPU)
        prev_cumulative = estimated_cumulative_phases[-1]
        current_cumulative_phase = prev_cumulative + delta_phi # Apply correction based on measurement
        estimated_cumulative_phases.append(current_cumulative_phase)

        # Apply phase correction (CPU) using original CPU chunk data
        corrected_chunk_cpu = curr_chunk_cpu * np.exp(-1j * current_cumulative_phase)
        corrected_chunks.append(corrected_chunk_cpu) # Store corrected CPU array

    logger.info("--- Improved Phase Correlation Complete ---")
    # Ensure all returned chunks are numpy arrays (they should be)
    return corrected_chunks, estimated_cumulative_phases


# --- WPD function updated with logging ---
def correct_phase_wpd(chunks, wavelet='db4', level=4):
    """
    Performs intra-chunk phase detrending using Wavelet Packet Decomposition (CPU).
    """
    logger.info(f"Applying Wavelet-Based Phase Correction (Wavelet: {wavelet}, Level: {level})")
    wpd_corrected_chunks = []
    try:
        wavelet_obj = pywt.Wavelet(wavelet)
    except ValueError as e: logger.error(f"Invalid wavelet: {e}"); return chunks
    except Exception as e: logger.error(f"Wavelet init error: {e}"); return chunks

    for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="WPD Processing"):
        chunk_np = np.asarray(chunk)
        if len(chunk_np) <= 2: wpd_corrected_chunks.append(chunk_np); continue
        try:
            max_level = pywt.dwt_max_level(len(chunk_np), wavelet_obj); actual_level = min(level, max_level)
            if level > max_level: logger.warning(f"WPD Lvl {level}>{max_level} Chk {i}. Using {actual_level}.")
            if actual_level <= 0: logger.warning(f"WPD Lvl {actual_level} invalid Chk {i}."); wpd_corrected_chunks.append(chunk_np); continue

            wp_real = pywt.WaveletPacket(data=np.real(chunk_np), wavelet=wavelet, mode='symmetric', maxlevel=actual_level)
            wp_imag = pywt.WaveletPacket(data=np.imag(chunk_np), wavelet=wavelet, mode='symmetric', maxlevel=actual_level)
            nodes_real = wp_real.get_level(actual_level, 'natural')

            for node_real in nodes_real:
                try:
                    node_path = node_real.path
                    if not isinstance(node_path, str) or node_path not in wp_imag: continue
                    node_imag = wp_imag[node_path]
                    real_data = np.array(node_real.data); imag_data = np.array(node_imag.data)
                    coeffs = real_data + 1j * imag_data
                    if len(coeffs) > 1:
                        inst_phase = np.unwrap(np.angle(coeffs + 1e-30)); x = np.arange(len(inst_phase)); A = np.vstack([x, np.ones_like(x)]).T
                        try:
                            slope, intercept = np.linalg.lstsq(A, inst_phase, rcond=None)[0]
                            linear_phase = intercept + slope * x; coeffs_corrected = coeffs * np.exp(-1j * linear_phase)
                            node_real.data = coeffs_corrected.real; node_imag.data = coeffs_corrected.imag
                        except np.linalg.LinAlgError: logger.debug(f"LinAlgError node {node_path} Chk {i}")
                except Exception as node_e: logger.debug(f"Node error {node_path} Chk {i}: {node_e}")

            corrected_real = wp_real.reconstruct(update=False); corrected_imag = wp_imag.reconstruct(update=False)
            corrected_chunk = (corrected_real + 1j * corrected_imag).astype(np.complex128)
            if len(corrected_chunk) != len(chunk_np): # Length correction
                 logger.debug(f"WPD Len Corr Chk {i}: {len(corrected_chunk)} vs {len(chunk_np)}")
                 corrected_chunk = corrected_chunk[:len(chunk_np)]
                 if len(corrected_chunk) < len(chunk_np): corrected_chunk = np.pad(corrected_chunk, (0, len(chunk_np) - len(corrected_chunk)), 'constant')
            chunk_rms = np.sqrt(np.mean(np.abs(chunk_np)**2)); corrected_rms = np.sqrt(np.mean(np.abs(corrected_chunk)**2))
            if chunk_rms > 1e-12 and corrected_rms > 1e-12: corrected_chunk *= (chunk_rms / corrected_rms)
            wpd_corrected_chunks.append(corrected_chunk)
        except Exception as e: logger.error(f"WPD Error Chk {i}: {e}", exc_info=False); wpd_corrected_chunks.append(chunk_np)
    logger.info("WPD Phase Correction Complete")
    return wpd_corrected_chunks