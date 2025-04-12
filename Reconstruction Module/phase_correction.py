# phase_correction.py
"""Functions for correcting phase discontinuities between chunks using GPU."""

import numpy as np
from tqdm import tqdm
import pywt
import sys
import logging
from . import log_config

# Setup logging for this module
log_config.setup_logging(level=logging.DEBUG, log_dir="run_logs")
logger = logging.getLogger(__name__)

# --- CuPy Import and Check ---
try:
    import cupy as cp
    # Check if a GPU is available and select the default one
    try:
        cp.cuda.Device(0).use()
        cupy_available = True
        logger.info("CuPy found and GPU available. Using GPU acceleration for correlation.")
    except cp.cuda.runtime.CUDARuntimeError as e:
        logger.error(f"CuPy found, but GPU unavailable/error: {e}. CANNOT PERFORM GPU CORRELATION.")
        cupy_available = False
except ImportError:
    logger.error("CuPy not found. CANNOT PERFORM GPU CORRELATION.")
    cupy_available = False
# --- End CuPy Import ---


def correct_phase_correlation(chunks, sdr_rate, recon_rate, base_overlap_factor, min_overlap_factor, max_lag_fraction, max_lag_abs):
    logger.info("\n--- Performing Improved Phase Correlation (GPU Required) ---")

    if not cupy_available:
        logger.critical("FATAL: Skipping phase correlation as GPU/CuPy is not available.")
        return chunks, [0.0] * len(chunks) # Return originals, cannot proceed

    if not chunks:
        logger.error("Error: No chunks provided for phase correlation.")
        return [], []

    # Calculate overlap samples (on CPU)
    chunk_duration_s = 0
    if len(chunks) > 0 and recon_rate > 0:
         chunk_duration_s = len(chunks[0]) / recon_rate
    else:
         logger.warning("Warning: Cannot determine chunk duration for overlap calc.")

    effective_overlap = max(min_overlap_factor, base_overlap_factor)
    overlap_samples_recon = int(round(chunk_duration_s * effective_overlap * recon_rate)) if recon_rate is not None else 0
    logger.info(f"Enhanced Overlap samples for Correlation: {overlap_samples_recon}")

    # Ensure first chunk is numpy array on CPU
    corrected_chunks = [np.asarray(chunks[0])]
    estimated_cumulative_phases = [0.0]

    # Define GPU variables list for cleanup
    gpu_vars_to_clean = ['prev_segment_gpu', 'curr_segment_gpu', 'xcorr_mag_gpu',
                         'xcorr_slice_gpu', 'best_lag_idx_in_search_gpu',
                         'aligned_curr_gpu', 'aligned_prev_gpu',
                         'complex_prod_gpu', 'phasors_gpu', 'avg_phasor_gpu',
                         'delta_phi_gpu'] # Add all potential GPU arrays

    for i in tqdm(range(1, len(chunks)), desc="GPU Phase Correlation"):
        prev_chunk_cpu = np.asarray(corrected_chunks[i-1])
        curr_chunk_cpu = np.asarray(chunks[i])

        if overlap_samples_recon <= 0 or len(prev_chunk_cpu) < overlap_samples_recon or len(curr_chunk_cpu) < overlap_samples_recon:
            status = f"overlap samples={overlap_samples_recon}" if overlap_samples_recon <= 0 else f"lengths prev={len(prev_chunk_cpu)}, curr={len(curr_chunk_cpu)}"
            logger.warning(f"Insufficient overlap ({status}) for chunk {i}. Skipping correlation, applying previous phase correction (CPU).")
            estimated_cumulative_phases.append(estimated_cumulative_phases[-1])
            corrected_chunk_cpu = curr_chunk_cpu * np.exp(-1j * estimated_cumulative_phases[-1])
            corrected_chunks.append(corrected_chunk_cpu)
            continue

        prev_segment_cpu = prev_chunk_cpu[-overlap_samples_recon:]
        curr_segment_cpu = curr_chunk_cpu[:overlap_samples_recon]

        delta_phi = 0.0
        best_lag = 0

        # --- GPU Processing Block ---
        # Define GPU variables within try block scope
        prev_segment_gpu, curr_segment_gpu = None, None
        aligned_curr_gpu, aligned_prev_gpu = None, None
        # ... other GPU vars initialized implicitly or set to None

        try:
            # Transfer data to GPU
            prev_segment_gpu = cp.asarray(prev_segment_cpu)
            curr_segment_gpu = cp.asarray(curr_segment_cpu)

            # Compute cross-correlation of magnitudes on GPU
            max_lag = min(max_lag_abs, int(overlap_samples_recon * max_lag_fraction))
            xcorr_mag_gpu = cp.correlate(cp.abs(curr_segment_gpu), cp.abs(prev_segment_gpu), mode='full')

            # Lag calculation
            n_curr = len(curr_segment_gpu)
            n_prev = len(prev_segment_gpu)
            lags = np.arange(-(n_curr - 1), n_prev) # CPU lags
            search_indices = np.where(np.abs(lags) <= max_lag)[0] # CPU indices

            if len(search_indices) > 0:
                valid_search_indices = search_indices[(search_indices >= 0) & (search_indices < len(xcorr_mag_gpu))]
                if len(valid_search_indices) > 0:
                    xcorr_slice_gpu = xcorr_mag_gpu[valid_search_indices]
                    best_lag_idx_in_search_gpu = cp.argmax(cp.abs(xcorr_slice_gpu))
                    best_lag_idx_in_search = int(best_lag_idx_in_search_gpu.get())
                    best_overall_idx = valid_search_indices[best_lag_idx_in_search]
                    best_lag = lags[best_overall_idx] # CPU lag value

            # Align segments on GPU
            if best_lag > 0:
                aligned_curr_gpu = curr_segment_gpu[best_lag:]
                aligned_prev_gpu = prev_segment_gpu[:len(aligned_curr_gpu)]
            elif best_lag < 0:
                aligned_prev_gpu = prev_segment_gpu[abs(best_lag):]
                aligned_curr_gpu = curr_segment_gpu[:len(aligned_prev_gpu)]
            else:
                aligned_curr_gpu = curr_segment_gpu
                aligned_prev_gpu = prev_segment_gpu

            # --- Estimate phase using Phasor Averaging ---
            if len(aligned_curr_gpu) > 0 and len(aligned_prev_gpu) > 0:
                min_len = min(len(aligned_curr_gpu), len(aligned_prev_gpu))
                # Calculate element-wise product on GPU
                complex_prod_gpu = aligned_curr_gpu[:min_len] * cp.conj(aligned_prev_gpu[:min_len])
                # Calculate magnitudes on GPU (add epsilon for stability)
                magnitudes_gpu = cp.abs(complex_prod_gpu) + 1e-20
                # Calculate phasors on GPU
                phasors_gpu = complex_prod_gpu / magnitudes_gpu
                # Average phasors on GPU
                avg_phasor_gpu = cp.mean(phasors_gpu)
                # Calculate angle of the average phasor on GPU, transfer to CPU
                if cp.abs(avg_phasor_gpu) > 1e-12: # Check magnitude of average
                    delta_phi_gpu = cp.angle(avg_phasor_gpu)
                    delta_phi = float(delta_phi_gpu.get()) # Get CPU float

        except Exception as corr_e:
             logger.error(f"ERROR during GPU correlation computation for chunk {i}: {corr_e}. Using delta_phi=0.")
             delta_phi = 0.0
             best_lag = 0
        finally:
             # --- Clear GPU memory ---
             # Use locals() and check type to safely delete CuPy arrays
             for var_name in gpu_vars_to_clean:
                 if var_name in locals() and isinstance(locals()[var_name], cp.ndarray):
                     del locals()[var_name]
             try:
                 mempool = cp.get_default_memory_pool()
                 mempool.free_all_blocks()
             except Exception as mem_e:
                 logger.warning(f"Error freeing CuPy memory pool: {mem_e}")
        # --- End GPU Processing Block ---

        # --- Diagnostic Log ---
        logger.info(f"  Chunk {i}: Est. Lag = {best_lag}, Delta Phi (Phasor Avg, deg) = {np.rad2deg(delta_phi):.4f}")

        # Update cumulative phase (CPU)
        prev_cumulative = estimated_cumulative_phases[-1]
        current_cumulative_phase = prev_cumulative + delta_phi
        estimated_cumulative_phases.append(current_cumulative_phase)

        # Apply phase correction (CPU)
        corrected_chunk_cpu = curr_chunk_cpu * np.exp(-1j * current_cumulative_phase)
        corrected_chunks.append(corrected_chunk_cpu)

    logger.info("--- Improved Phase Correlation Complete ---")
    return corrected_chunks, estimated_cumulative_phases

# Update WPD function to use logger
def correct_phase_wpd(chunks, wavelet, level):
    logger.info("\n--- Performing Wavelet-Based Phase Correction (Intra-Chunk Detrending) ---")
    wpd_corrected_chunks = []
    wavelet_obj = None
    try:
        wavelet_obj = pywt.Wavelet(wavelet)
    except Exception as e:
        logger.critical(f"Fatal Error: Could not initialize wavelet '{wavelet}': {e}. Skipping WPD.")
        return chunks # Return original chunks if wavelet fails

    for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="WPD Processing"):
        actual_max_level = 0
        if len(chunk) > 0:
            try:
                actual_max_level = pywt.dwt_max_level(len(chunk), wavelet_obj)
            except Exception: # Catch potential errors from dwt_max_level
                pass # Keep actual_max_level at 0

        if len(chunk) <= 2 or level > actual_max_level or level <= 0:
            # Add appropriate warnings if needed, similar to original script
            wpd_corrected_chunks.append(chunk)
            continue

        try:
            wp_real = pywt.WaveletPacket(data=np.real(chunk), wavelet=wavelet, mode='symmetric', maxlevel=level)
            wp_imag = pywt.WaveletPacket(data=np.imag(chunk), wavelet=wavelet, mode='symmetric', maxlevel=level)
            nodes_real = wp_real.get_level(level, 'natural')

            for node_real in nodes_real:
                try:
                    current_node_path = node_real.path
                    if not isinstance(current_node_path, str): continue # Skip if path is not string

                    if current_node_path not in wp_imag: continue # Skip if no matching imag node
                    node_imag = wp_imag[current_node_path]

                    real_data = np.array(node_real.data)
                    imag_data = np.array(node_imag.data)
                    coeff_complex = real_data + 1j * imag_data

                    if len(coeff_complex) > 1:
                        inst_phase = np.unwrap(np.angle(coeff_complex + 1e-30))
                        x = np.arange(len(inst_phase))
                        try:
                            A = np.vstack([x, np.ones(len(x))]).T
                            phase_slope, phase_intercept = np.linalg.lstsq(A, inst_phase, rcond=None)[0]
                            linear_phase_trend = phase_intercept + phase_slope * x
                            phase_correction_factor = np.exp(-1j * linear_phase_trend)
                            corrected_coeffs = coeff_complex * phase_correction_factor
                            node_real.data = corrected_coeffs.real
                            node_imag.data = corrected_coeffs.imag
                        except np.linalg.LinAlgError: pass # Ignore LinAlgErrors during lstsq
                        except Exception: pass # Ignore other phase trend errors silently for now
                except Exception: pass # Ignore errors processing individual nodes

            corrected_real = wp_real.reconstruct(update=False)
            corrected_imag = wp_imag.reconstruct(update=False)
            corrected_chunk = (corrected_real + 1j * corrected_imag).astype(np.complex128)

            # Ensure length consistency
            if len(corrected_chunk) != len(chunk):
                 corrected_chunk = corrected_chunk[:len(chunk)]
                 if len(corrected_chunk) < len(chunk):
                      pad_len = len(chunk) - len(corrected_chunk)
                      corrected_chunk = np.pad(corrected_chunk, (0, pad_len), 'constant')

            # Maintain RMS
            chunk_rms = np.sqrt(np.mean(np.abs(chunk)**2))
            corrected_rms = np.sqrt(np.mean(np.abs(corrected_chunk)**2))
            if chunk_rms > 1e-12 and corrected_rms > 1e-12:
                corrected_chunk *= (chunk_rms / corrected_rms)
            wpd_corrected_chunks.append(corrected_chunk)

        except Exception as e: # Catch errors during WPD setup/reconstruction
            logger.error(f"General Error in WPD processing chunk {i}: {e}")
            wpd_corrected_chunks.append(chunk) # Use original on error

    logger.info("--- WPD Phase Correction (Intra-Chunk Detrending) Complete ---")
    # Ensure return is list
    return wpd_corrected_chunks if isinstance(wpd_corrected_chunks, list) else chunks