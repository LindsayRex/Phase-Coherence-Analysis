# pilot_phase_correction.py
"""Corrects phase discontinuities using pilot tone and optional WPD."""

import numpy as np
from tqdm import tqdm
import pywt
import sys
import logging
from . import log_config

# Setup logging for this module
log_config.setup_logging(level=logging.DEBUG, log_dir="run_logs")
logger = logging.getLogger(__name__)

cupy_available = False # Assume False initially
try:
    import cupy as cp
    try:
        cp.cuda.Device(0).use(); cupy_available = True; logger.info("CuPy/GPU available.")
    except: logger.warning("CuPy found, but GPU unavailable. Using CPU.")
except ImportError: logger.warning("CuPy not found. Using CPU.")
# ---

def extract_pilot_phase(signal_segment, fs, pilot_bb_freq, n_fft=None, n_fft_factor=1):
    """
    Extracts the phase of a pilot tone using FFT with increased resolution.

    Args:
        signal_segment (np.ndarray or cp.ndarray): IQ data segment.
        fs (float): Sample rate (Hz).
        pilot_bb_freq (float): Expected BASEBAND frequency offset (Hz).
        n_fft (int, optional): Explicit FFT length. If None, calculated based on N * n_fft_factor.
        n_fft_factor (int): Factor to multiply N by if n_fft is None (kept for context).

    Returns:
        float: Estimated phase in radians, or np.nan if error.
    """
    is_gpu_array = cupy_available and isinstance(signal_segment, cp.ndarray)
    xp = cp if is_gpu_array else np

    N = len(signal_segment)
    if N < 32:
        logger.debug(f"Segment too short ({N})")
        return np.nan

    # --- Determine FFT Length ---
    if n_fft is None:
        # Calculate based on factor if n_fft not explicitly provided
        n_fft_calc = N * n_fft_factor
        n_fft = n_fft_calc # Use the calculated length

    logger.debug(f"Using n_fft={n_fft} for segment length {N}")

    if n_fft < N:
        signal_segment = signal_segment[:n_fft] # Ensure segment matches n_fft if smaller

    try:
        # Apply Hann window (length N, before padding)
        window = xp.hanning(N) # Window original segment length
        # Apply window to original segment data before potential padding by FFT
        windowed_segment = signal_segment[:N] * window

        # Compute FFT with specified length n_fft (pads if n_fft > N)
        fft_result = xp.fft.fft(windowed_segment, n=n_fft)
        fft_freqs = xp.fft.fftfreq(n_fft, d=1/fs)

        # ... (rest of the function: find peak, validate, return phase) ...
        pilot_bin_index = xp.argmin(xp.abs(fft_freqs - pilot_bb_freq))
        freq_at_bin = fft_freqs[pilot_bin_index]
        freq_resolution = fs / n_fft
        freq_diff = xp.abs(freq_at_bin - pilot_bb_freq)
        freq_tolerance = 5.0 * freq_resolution

        logger.info(f"Pilot FFT (N={n_fft}): Peak at {freq_at_bin/1e6:.4f} MHz, Expected {pilot_bb_freq/1e6:.4f} MHz (Res: {freq_resolution/1e3:.2f} kHz)")

        if freq_diff > freq_tolerance:
            logger.warning(f"Pilot peak frequency mismatch > tolerance ({freq_tolerance/1e3:.1f} kHz). Found at {freq_at_bin/1e6:.4f} MHz (Diff: {freq_diff/1e3:.1f} kHz)")
            return np.nan

        pilot_phase = xp.angle(fft_result[pilot_bin_index])
        return float(pilot_phase.get()) if is_gpu_array else float(pilot_phase)

    except Exception as e:
        logger.error(f"Error during pilot phase extraction: {e}", exc_info=False)
        return np.nan

def correct_phase_pilot_tone(chunks, metadata, global_attrs,
                             apply_wpd=False, wavelet='db4', wpd_level=4,
                             n_fft_factor_pilot=4):
    """
    Corrects chunk phases using pilot tone (with higher FFT resolution).
    Optionally applies WPD correction afterwards.

    Args:
        chunks (list): List of chunk data (NumPy complex128 arrays) AT SDR RATE.
        metadata (list): Metadata dictionaries for each chunk.
        global_attrs (dict): Global attributes dictionary.
        apply_wpd (bool): If True, run WPD correction after pilot correction.
        wavelet (str): Wavelet name for WPD.
        wpd_level (int): Decomposition level for WPD.
        n_fft_factor_pilot (int): Factor to increase FFT length for pilot extraction.

    Returns:
        tuple: (corrected_chunks, estimated_cumulative_phases)
    """
    logger.info("Performing Phase Correction using Pilot Tone (High FFT Res)...")

    if not cupy_available:
        logger.warning("CuPy/GPU not available, pilot extraction on CPU.")
        xp = np
    else:
        xp = cp

    if not chunks or not metadata or len(chunks) != len(metadata):
        logger.error("Invalid chunks/metadata")
        return chunks, [0.0] * len(chunks)

    pilot_added = global_attrs.get('pilot_tone_added', False)
    pilot_bb_offset_hz = global_attrs.get('pilot_offset_hz', None)
    if not pilot_added or pilot_bb_offset_hz is None:
        logger.warning("Pilot metadata missing")
        return chunks, [0.0] * len(chunks)

    logger.info(f"Using pilot tone baseband offset: {pilot_bb_offset_hz / 1e6:.3f} MHz")

    fs = metadata[0].get('sdr_sample_rate_hz') # Use SDR rate passed in metadata
    if fs is None:
        logger.error("SDR sample rate missing")
        return chunks, [0.0] * len(chunks)

    overlap_factor = global_attrs.get('overlap_factor', 0.1)
    chunk_duration_s = metadata[0].get('intended_duration_s')
    if chunk_duration_s is None:
        logger.warning(f"intended_duration_s missing")
        overlap_samples = int(round(len(chunks[0]) * overlap_factor)) if len(chunks) > 0 else 0
    else:
        overlap_samples = int(round(chunk_duration_s * overlap_factor * fs))

    overlap_samples_fft = overlap_samples # Use actual overlap samples
    n_fft_pilot = overlap_samples_fft * n_fft_factor_pilot # Calculate FFT length

    logger.info(f"Overlap samples: {overlap_samples_fft}, Pilot FFT Length: {n_fft_pilot}")
    if overlap_samples_fft < 32:
        logger.warning(f"Overlap too small")
        return chunks, [0.0] * len(chunks)

    estimated_cumulative_phases = [0.0] * len(chunks)
    last_pilot_phase = np.nan
    corrected_chunks_list = [None] * len(chunks)
    corrected_chunks_list[0] = np.asarray(chunks[0])

    for i in tqdm(range(1, len(chunks)), desc="Pilot Phase Correction (SDR Rate)"):
        prev_chunk_original = np.asarray(chunks[i-1]) # Use original SDR rate chunks
        curr_chunk_original = np.asarray(chunks[i])

        if len(prev_chunk_original) < overlap_samples_fft or len(curr_chunk_original) < overlap_samples_fft:
            logger.warning(f"Insufficient FFT overlap chunk {i}. Applying previous correction.")
            estimated_cumulative_phases[i] = estimated_cumulative_phases[i-1]
            last_pilot_phase = np.nan
            corrected_chunks_list[i] = curr_chunk_original # Store original if skipped
            continue

        prev_overlap_cpu = prev_chunk_original[-overlap_samples_fft:]
        curr_overlap_cpu = curr_chunk_original[:overlap_samples_fft]
        prev_overlap_dev = xp.asarray(prev_overlap_cpu)
        curr_overlap_dev = xp.asarray(curr_overlap_cpu)

        # Extract phases using specified n_fft_pilot length
        phase_prev_end = extract_pilot_phase(prev_overlap_dev, fs, pilot_bb_offset_hz, n_fft=n_fft_pilot)
        phase_curr_start = extract_pilot_phase(curr_overlap_dev, fs, pilot_bb_offset_hz, n_fft=n_fft_pilot)

        delta_phi = 0.0
        if not np.isnan(phase_prev_end) and not np.isnan(phase_curr_start):
            delta_phi = phase_curr_start - phase_prev_end
            delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
            logger.info(f"  Chunk {i}: PrevEnd Ph={np.rad2deg(phase_prev_end):.2f}, CurrStart Ph={np.rad2deg(phase_curr_start):.2f}, Delta Phi={np.rad2deg(delta_phi):.4f} deg")
        else:
            logger.warning(f"Pilot phase extraction failed boundary {i}. delta_phi=0.")
            delta_phi = 0.0

        estimated_cumulative_phases[i] = estimated_cumulative_phases[i-1] - delta_phi
        last_pilot_phase = phase_curr_start

        if cupy_available:
            del prev_overlap_dev, curr_overlap_dev
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    logger.info("Applying calculated pilot tone phase corrections...")
    for i in range(1, len(chunks)):
        chunk_np = np.asarray(chunks[i])
        phase_correction_factor = np.exp(-1j * estimated_cumulative_phases[i])
        corrected_chunks_list[i] = chunk_np * phase_correction_factor # Correct original chunk

    if apply_wpd:
        logger.info("Applying WPD after pilot tone correction...")
        corrected_chunks_list = correct_phase_wpd(corrected_chunks_list, wavelet=wavelet, level=wpd_level)

    logger.info("Phase Correction steps complete.")
    return corrected_chunks_list, estimated_cumulative_phases

def correct_phase_wpd(chunks, wavelet='db4', level=4):
    logger.info(f"Applying Wavelet-Based Phase Correction (Wavelet: {wavelet}, Level: {level})")
    wpd_corrected_chunks = []
    try:
        wavelet_obj = pywt.Wavelet(wavelet)
    except Exception as e:
        logger.error(f"Cannot initialize wavelet '{wavelet}': {e}. Skipping WPD.")
        return chunks

    for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="WPD Processing"):
        chunk_np = np.asarray(chunk)
        if len(chunk_np) <= 2:
            wpd_corrected_chunks.append(chunk_np)
            continue

        try:
            max_level = pywt.dwt_max_level(len(chunk_np), wavelet_obj)
            actual_level = min(level, max_level)
            if level > max_level:
                logger.warning(f"WPD level {level} > max {max_level} for chunk {i}. Using {actual_level}.")
            if actual_level <= 0:
                logger.warning(f"WPD level {actual_level} invalid for chunk {i}. Skipping.")
                wpd_corrected_chunks.append(chunk_np)
                continue

            wp_real = pywt.WaveletPacket(data=np.real(chunk_np), wavelet=wavelet, mode='symmetric', maxlevel=actual_level)
            wp_imag = pywt.WaveletPacket(data=np.imag(chunk_np), wavelet=wavelet, mode='symmetric', maxlevel=actual_level)
            nodes_real = wp_real.get_level(actual_level, 'natural')

            for node_real in nodes_real:
                try:
                    node_path = node_real.path
                    if not isinstance(node_path, str) or node_path not in wp_imag:
                        continue
                    node_imag = wp_imag[node_path]
                    real_data = np.array(node_real.data)
                    imag_data = np.array(node_imag.data)
                    coeffs = real_data + 1j * imag_data
                    if len(coeffs) > 1:
                        inst_phase = np.unwrap(np.angle(coeffs + 1e-30))
                        x = np.arange(len(inst_phase))
                        A = np.vstack([x, np.ones_like(x)]).T
                        try:
                            slope, intercept = np.linalg.lstsq(A, inst_phase, rcond=None)[0]
                            linear_phase = intercept + slope * x
                            coeffs_corrected = coeffs * np.exp(-1j * linear_phase)
                            node_real.data = coeffs_corrected.real
                            node_imag.data = coeffs_corrected.imag
                        except np.linalg.LinAlgError:
                            pass
                except Exception:
                    pass

            corrected_real = wp_real.reconstruct(update=False)
            corrected_imag = wp_imag.reconstruct(update=False)
            corrected_chunk = (corrected_real + 1j * corrected_imag).astype(np.complex128)
            if len(corrected_chunk) != len(chunk_np):
                corrected_chunk = corrected_chunk[:len(chunk_np)]
                if len(corrected_chunk) < len(chunk_np):
                    corrected_chunk = np.pad(corrected_chunk, (0, len(chunk_np) - len(corrected_chunk)), 'constant')

            chunk_rms = np.sqrt(np.mean(np.abs(chunk_np)**2))
            corrected_rms = np.sqrt(np.mean(np.abs(corrected_chunk)**2))
            if chunk_rms > 1e-12 and corrected_rms > 1e-12:
                corrected_chunk *= (chunk_rms / corrected_rms)

            wpd_corrected_chunks.append(corrected_chunk)
        except Exception as e:
            logger.error(f"Error during WPD chunk {i}: {e}", exc_info=False)
            wpd_corrected_chunks.append(chunk_np)

    logger.info("WPD Phase Correction Complete")
    return wpd_corrected_chunks