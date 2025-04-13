# pilot_phase_correction.py
"""
Performs phase correction using Wavelet Packet Decomposition based alignment
and optionally removes the pilot tone using a notch filter.
Optimized for GPU acceleration where available (CuPy/cuSignal).
Removes old/fallback logic, focusing on the WPD-based approach.
"""

import numpy as np
from tqdm import tqdm
import pywt
import sys
import logging
from scipy.signal import iirnotch, filtfilt # Using SciPy filtfilt for CPU/fallback

# Import config settings directly
from . import config

# Setup logging for this module
logger = logging.getLogger(__name__)

# --- CuPy/cuSignal Import and Check ---
cupy_available = False
cusignal_available = False
try:
    import cupy as cp
    try:
        cp.cuda.Device(0).use() # Check GPU availability
        cupy_available = True
        try:
            import cusignal # Check for cuSignal
            cusignal_available = True
            logger.info("CuPy/GPU and cuSignal available. Enabling GPU acceleration.")
        except ImportError:
            logger.info("CuPy/GPU available but cuSignal not found. Notch filter on CPU.")
    except Exception as e_gpu:
        logger.warning(f"CuPy found, but GPU unavailable/error: {e_gpu}. Processing on CPU.")
except ImportError:
    logger.warning("CuPy not found. Processing on CPU using NumPy.")
# --- End CuPy/cuSignal Import ---


def _determine_wpd_pilot_path(sdr_rate, pilot_freq, overlap_samples, wpd_wavelet, wpd_level):
    """Helper to find the WPD node path closest to the pilot frequency."""
    try:
        freq_res = sdr_rate / (2**(wpd_level + 1)) # Approx center freq resolution
        # Dummy decomposition on CPU to get node paths
        wp_dummy = pywt.WaveletPacket(data=np.zeros(overlap_samples), wavelet=wpd_wavelet, mode='symmetric', maxlevel=wpd_level)
        nodes = wp_dummy.get_level(wpd_level, 'freq')
        # Calculate approximate center frequency for each terminal node
        node_freqs = []
        valid_paths = []
        for node in nodes:
             # Check if it's a terminal node at the desired level
             if node.path and len(node.path) == wpd_level:
                 # Calculate frequency based on path ('a'=0, 'd'=1 -> binary)
                 # This is an approximation, actual coverage depends on wavelet filters
                 try:
                     binary_repr = node.path.replace('a','0').replace('d','1')
                     node_index = int(binary_repr, 2)
                     # Center frequency approx based on index and resolution
                     # This assumes standard frequency ordering, adjust if needed for pywt
                     center_freq = node_index * freq_res + freq_res / 2.0 # Midpoint of bin
                     node_freqs.append(center_freq)
                     valid_paths.append(node.path)
                 except ValueError:
                      logger.debug(f"Could not interpret node path '{node.path}' as binary.")
                      continue # Skip nodes with unexpected paths

        if not node_freqs:
            logger.error("Could not extract node frequencies from dummy WPD.")
            return None, np.nan

        node_freqs = np.array(node_freqs)
        # Find the path corresponding to the minimum absolute difference
        pilot_bin_idx = np.argmin(np.abs(node_freqs - pilot_freq))
        pilot_subband_path = valid_paths[pilot_bin_idx]
        actual_bin_freq = node_freqs[pilot_bin_idx]
        logger.info(f"Target Pilot Freq: {pilot_freq/1e6:.3f} MHz. Using WPD node: '{pilot_subband_path}' near {actual_bin_freq/1e6:.3f} MHz (Res: {freq_res/1e6:.3f} MHz)")
        return pilot_subband_path, actual_bin_freq
    except Exception as e:
        logger.error(f"Error determining WPD pilot subband: {e}", exc_info=True)
        return None, np.nan


def perform_phase_correction(chunks_in, metadata, global_attrs):
    """
    Main function to perform phase correction and pilot removal.

    Args:
        chunks_in (list): List of chunk data (NumPy complex128 arrays) AT SDR RATE.
        metadata (list): Metadata dictionaries for each chunk.
        global_attrs (dict): Global attributes dictionary.

    Returns:
        tuple: (processed_chunks, estimated_cumulative_phases)
               - processed_chunks (list): List of NumPy arrays (CPU) after processing.
               - estimated_cumulative_phases (list): Estimated cumulative phase offset (CPU floats).
    """
    logger.info("===== Phase Correction & Pilot Removal Module =====")
    xp = cp if cupy_available else np
    use_gpu = cupy_available
    backend_name = "GPU (CuPy)" if use_gpu else "CPU (NumPy)"
    logger.info(f"Using backend: {backend_name}")

    # --- Input Validation ---
    if not chunks_in or not metadata or len(chunks_in) != len(metadata):
        logger.error("Invalid chunks/metadata. Returning empty lists.")
        return [], []
    if not global_attrs:
        logger.error("Global attributes missing. Returning original chunks.")
        return [np.asarray(c) for c in chunks_in], [0.0] * len(chunks_in)

    num_chunks = len(chunks_in)
    chunks_cpu = [np.asarray(c, dtype=np.complex128) for c in chunks_in] # Work with CPU arrays initially
    estimated_cumulative_phases = [0.0] * num_chunks
    processed_chunks_list = [chunks_cpu[0]] # Start with chunk 0 on CPU

    # --- Phase Alignment (WPD-based) ---
    if config.APPLY_WAVELET_PHASE_ALIGNMENT:
        logger.info("--- Applying Wavelet Phase Alignment ---")
        sdr_rate = global_attrs.get('sdr_sample_rate_hz')
        pilot_freq = global_attrs.get('pilot_offset_hz')
        overlap_factor = global_attrs.get('overlap_factor', 0.1)
        wpd_wavelet = config.WPD_WAVELET_ALIGN
        wpd_level = config.WPD_LEVEL_ALIGN

        if sdr_rate is None or pilot_freq is None:
            logger.error("Missing SDR rate or pilot freq for WPD alignment. Skipping alignment.")
            processed_chunks_list = chunks_cpu # Keep original if skipping
        else:
            chunk_len = len(chunks_cpu[0]) if num_chunks > 0 else 0
            overlap_samples = int(chunk_len * overlap_factor)
            logger.info(f"WPD Alignment: Wavelet='{wpd_wavelet}', Level={wpd_level}, Overlap={overlap_samples}")

            if overlap_samples < 32:
                logger.warning(f"Overlap ({overlap_samples}) too small. Skipping alignment.")
                processed_chunks_list = chunks_cpu
            else:
                pilot_subband_path, _ = _determine_wpd_pilot_path(sdr_rate, pilot_freq, overlap_samples, wpd_wavelet, wpd_level)

                if pilot_subband_path is None:
                    logger.error("Failed to find pilot subband. Skipping alignment.")
                    processed_chunks_list = chunks_cpu
                else:
                    # --- Alignment Loop ---
                    phase_deltas_log = []
                    for i in tqdm(range(1, num_chunks), desc="WPD Phase Align"):
                        prev_chunk_cpu = processed_chunks_list[-1] # Prev corrected (CPU)
                        curr_chunk_original_cpu = chunks_cpu[i]    # Current original (CPU)

                        delta_phi = 0.0
                        # Ensure lengths are sufficient
                        if len(prev_chunk_cpu) < overlap_samples or len(curr_chunk_original_cpu) < overlap_samples:
                             logger.warning(f"Chunk {i}: Insufficient overlap length. delta_phi=0.")
                        else:
                             # Extract overlaps (CPU for pywt)
                             prev_overlap_cpu = prev_chunk_cpu[-overlap_samples:]
                             curr_overlap_cpu = curr_chunk_original_cpu[:overlap_samples]
                             try:
                                 # Perform WPD (CPU)
                                 wp_prev = pywt.WaveletPacket(data=prev_overlap_cpu, wavelet=wpd_wavelet, mode='symmetric', maxlevel=wpd_level)
                                 wp_curr = pywt.WaveletPacket(data=curr_overlap_cpu, wavelet=wpd_wavelet, mode='symmetric', maxlevel=wpd_level)
                                 prev_coeffs = wp_prev[pilot_subband_path].data
                                 curr_coeffs = wp_curr[pilot_subband_path].data

                                 if len(prev_coeffs) > 0 and len(curr_coeffs) > 0:
                                     # Phase calc (CPU)
                                     prev_phase_unwrapped = np.unwrap(np.angle(prev_coeffs.astype(np.complex128)))
                                     curr_phase_unwrapped = np.unwrap(np.angle(curr_coeffs.astype(np.complex128)))
                                     delta_phi = np.mean(curr_phase_unwrapped - prev_phase_unwrapped)
                                 else: logger.warning(f"Chunk {i}: Empty pilot coeffs. delta_phi=0.")
                             except Exception as e: logger.error(f"Chunk {i} WPD/Phase Error: {e}. delta_phi=0.", exc_info=False)

                        phase_deltas_log.append(np.rad2deg(delta_phi))
                        logger.info(f"  Chunk {i}: Delta Phi = {np.rad2deg(delta_phi):.4f} deg")

                        # Update & Apply Cumulative Phase Correction
                        cumulative_phase = estimated_cumulative_phases[i-1] + delta_phi
                        estimated_cumulative_phases[i] = cumulative_phase
                        try:
                            # Apply correction using chosen backend
                            curr_chunk_dev = xp.asarray(curr_chunk_original_cpu) # Move original to device
                            correction_factor = xp.exp(-1j * cumulative_phase)
                            corrected_chunk_dev = curr_chunk_dev * correction_factor

                            # Store result as CPU array
                            processed_chunks_list.append(cp.asnumpy(corrected_chunk_dev) if use_gpu else corrected_chunk_dev)

                            # Clean up GPU mem for this iter
                            if use_gpu: del curr_chunk_dev, correction_factor, corrected_chunk_dev; cp.get_default_memory_pool().free_all_blocks()
                        except Exception as e:
                            logger.error(f"Chunk {i} Correction Error: {e}. Appending original.", exc_info=False)
                            processed_chunks_list.append(curr_chunk_original_cpu) # Append original on error
                            estimated_cumulative_phases[i] = estimated_cumulative_phases[i-1] # Revert phase estimate

                    logger.info(f"Phase deltas applied (deg): {phase_deltas_log}")
                    logger.info("--- Wavelet Phase Alignment Complete ---")
    else:
        logger.info("Skipping Wavelet Phase Alignment (disabled in config).")
        processed_chunks_list = chunks_cpu # Keep original CPU arrays


    # --- Pilot Tone Removal Step ---
    if config.APPLY_PILOT_REMOVAL:
        logger.info("--- Applying Pilot Tone Removal ---")
        sdr_rate = global_attrs.get('sdr_sample_rate_hz')
        pilot_freq = global_attrs.get('pilot_offset_hz')
        notch_q = config.PILOT_NOTCH_Q

        if sdr_rate is None or pilot_freq is None:
            logger.error("Missing rate/freq for pilot removal. Skipping.")
        else:
            try:
                # Design notch filter coefficients (once on CPU)
                b_notch, a_notch = iirnotch(pilot_freq, Q=notch_q, fs=sdr_rate)
                b_notch = b_notch.astype(np.float64)
                a_notch = a_notch.astype(np.float64)
                logger.info(f"Notch filter designed for {pilot_freq/1e6:.3f} MHz (Q={notch_q})")

                # Apply filtering using appropriate backend
                chunks_to_filter = [ensure_array(chunk, use_gpu=use_gpu) for chunk in processed_chunks_list]
                
                filtered_chunks_device = []
                if use_gpu and cusignal_available:
                    logger.info("Attempting pilot removal with cuSignal filtfilt...")
                    b_gpu = cp.asarray(b_notch)
                    a_gpu = cp.asarray(a_notch)
                    all_successful = True
                    for i, chunk_dev in tqdm(enumerate(chunks_to_filter), total=num_chunks, desc="GPU Notch Filtering"):
                        if len(chunk_dev) > max(len(a_gpu), len(b_gpu)) * 3:
                             try:
                                 filtered_chunk_dev = cusignal.filtfilt(b_gpu, a_gpu, chunk_dev)
                                 filtered_chunks_device.append(filtered_chunk_dev)
                             except Exception as e:
                                 logger.error(f"GPU filtfilt error chunk {i}: {e}. Will fallback to CPU for ALL chunks.", exc_info=False)
                                 all_successful = False
                                 break # Exit loop on first error, fallback strategy below
                        else:
                            logger.warning(f"Chunk {i} too short for filtfilt. Keeping unfiltered.")
                            filtered_chunks_device.append(chunk_dev) # Keep original if too short

                    if not all_successful:
                        logger.warning("Fallback to CPU filtfilt due to GPU errors.")
                        use_gpu = False # Force CPU path below
                    else:
                        processed_chunks_list = filtered_chunks_device # Keep results on GPU for now if successful

                # If not using GPU or fallback needed
                if not use_gpu or not cusignal_available:
                    logger.info("Applying pilot removal with SciPy filtfilt on CPU...")
                    filtered_chunks_cpu = []
                    for i, chunk_dev in tqdm(enumerate(chunks_to_filter), total=num_chunks, desc="CPU Notch Filtering"):
                        # Ensure chunk is on CPU
                        chunk_cpu = cp.asnumpy(chunk_dev) if cupy_available and isinstance(chunk_dev, cp.ndarray) else chunk_dev
                        if len(chunk_cpu) > max(len(a_notch), len(b_notch)) * 3:
                            try:
                                filtered_chunk_cpu = filtfilt(b_notch, a_notch, chunk_cpu)
                                filtered_chunks_cpu.append(filtered_chunk_cpu.astype(np.complex128))
                            except Exception as e:
                                logger.error(f"CPU filtfilt error chunk {i}: {e}. Appending original.", exc_info=False)
                                filtered_chunks_cpu.append(chunk_cpu) # Append original chunk on error
                        else:
                            logger.warning(f"Chunk {i} too short for filtfilt. Appending original.")
                            filtered_chunks_cpu.append(chunk_cpu)
                    processed_chunks_list = filtered_chunks_cpu # Update list with CPU results

                logger.info("--- Pilot Tone Removal Complete ---")

            except ValueError as ve:
                logger.error(f"Error designing notch filter (invalid freq/Q?): {ve}. Skipping removal.", exc_info=True)
            except Exception as e:
                logger.error(f"Error during pilot removal setup: {e}. Skipping removal.", exc_info=True)
    else:
        logger.info("Skipping Pilot Tone Removal (disabled in config).")

    # --- Final Conversion and Return ---
    # Ensure all results are returned as NumPy arrays on CPU
    final_cpu_chunks = []
    for chunk in processed_chunks_list:
        if cupy_available and isinstance(chunk, cp.ndarray):
            final_cpu_chunks.append(cp.asnumpy(chunk))
        else:
            final_cpu_chunks.append(np.asarray(chunk)) # Ensure it's numpy

    if len(final_cpu_chunks) != num_chunks:
        logger.error("Length mismatch in final processed chunks list! Returning original.")
        return [np.asarray(c) for c in chunks_in], [0.0] * len(chunks_in)

    # Final GPU memory cleanup
    if cupy_available:
        cp.get_default_memory_pool().free_all_blocks()

    logger.info("===== Phase Correction Module Finished =====")
    return final_cpu_chunks, estimated_cumulative_phases