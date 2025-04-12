# preprocessing.py
"""Functions for initial signal preprocessing steps."""

import numpy as np
from scipy import signal as sig
from tqdm import tqdm
import matplotlib.pyplot as plt

def scale_chunks(chunks, target_rms):
    """
    Scales each chunk to have a specific target RMS amplitude.

    Args:
        chunks (list): List of numpy arrays (complex128) containing IQ data.
        target_rms (float): The desired RMS value for each chunk.

    Returns:
        list: List of scaled numpy arrays (complex128). Returns None on critical error.
    """
    print("\n--- Correcting Initial Amplitude Scaling ---")
    scaled_chunks = []
    print("Chunk | RMS Before | Scaling Factor | RMS After  | Max Abs (After)")
    print("------|------------|----------------|------------|----------------")
    scaling_successful = True

    for i, chunk in enumerate(chunks):
        if len(chunk) == 0:
            scaled_chunks.append(chunk)
            print(f"{i:<5d} | --- EMPTY ---    | ---            | --- EMPTY ---    | ---")
            continue

        if not np.all(np.isfinite(chunk)):
            print(f"ERROR: Chunk {i} non-finite BEFORE scaling.")
            scaling_successful = False
            scaled_chunks.append(chunk) # Append original on error to maintain list length
            continue

        rms_before = np.sqrt(np.mean(np.abs(chunk)**2))
        max_abs_before = np.max(np.abs(chunk)) if len(chunk) > 0 else 0

        if rms_before < 1e-12:
            print(f"{i:<5d} | {rms_before:.4e}       | SKIPPED (Zero) | {rms_before:.4e}       | {max_abs_before:.4e}")
            scaled_chunks.append(chunk) # Append original if RMS is zero
            continue

        scaling_factor = target_rms / rms_before
        scaled_chunk = (chunk * scaling_factor).astype(np.complex128)

        if not np.all(np.isfinite(scaled_chunk)):
            print(f"ERROR: Chunk {i} non-finite AFTER scaling.")
            scaling_successful = False
            scaled_chunks.append(scaled_chunk) # Append erroneous chunk
            continue

        rms_after = np.sqrt(np.mean(np.abs(scaled_chunk)**2))
        max_abs_after = np.max(np.abs(scaled_chunk)) if len(scaled_chunk) > 0 else 0

        # Check if scaling achieved the target RMS within a tolerance
        if not np.isclose(rms_after, target_rms, rtol=5e-3): # Relaxed tolerance
            print(f"WARNING: Chunk {i} RMS scaling mismatch ({rms_after:.4e} vs {target_rms:.4e})")

        print(f"{i:<5d} | {rms_before:.4e} | {scaling_factor:<14.4f} | {rms_after:.4e} | {max_abs_after:.4e}")
        scaled_chunks.append(scaled_chunk)

    if not scaling_successful:
        print("\nERROR: Non-finite values detected during scaling. Proceeding, but results may be affected.")
        # Decide whether to return None or the partially processed list
        # return None # Hard fail
        # For now, return the list which might contain non-finite data
        pass

    print("--- Initial Amplitude Scaling Complete ---")
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
    print("\n--- Upsampling chunks (using polyphase filter) ---")
    upsampled_chunks_list = []

    if sdr_rate <= 0 or recon_rate <= 0:
        print("Error: Invalid sample rates for upsampling.")
        return [np.zeros_like(c) for c in chunks] # Return zero arrays matching input structure

    for i, chunk_data in tqdm(enumerate(chunks), total=len(chunks), desc="Upsampling"):
        if len(chunk_data) == 0:
             upsampled_chunks_list.append(chunk_data) # Append empty if input is empty
             continue

        # Calculate target length based on duration and new rate
        chunk_duration = len(chunk_data) / sdr_rate
        num_samples_chunk_recon = int(round(chunk_duration * recon_rate))

        if len(chunk_data) < 2:
            upsampled_chunks_list.append(np.zeros(num_samples_chunk_recon, dtype=complex))
            continue

        try:
            rms_in = np.sqrt(np.mean(np.abs(chunk_data)**2))
            # max_abs_in = np.max(np.abs(chunk_data)) if len(chunk_data)>0 else 0 # For debug

            if rms_in < 1e-12:
                upsampled_chunks_list.append(np.zeros(num_samples_chunk_recon, dtype=complex))
                continue

            # Calculate integer resampling factors for resample_poly
            common_divisor = np.gcd(int(recon_rate), int(sdr_rate))
            up_factor = int(recon_rate // common_divisor)
            down_factor = int(sdr_rate // common_divisor)

            # Upsample real and imaginary parts separately
            resampled_real = sig.resample_poly(chunk_data.real, up=up_factor, down=down_factor, window=('kaiser', 5.0))
            resampled_imag = sig.resample_poly(chunk_data.imag, up=up_factor, down=down_factor, window=('kaiser', 5.0))
            upsampled_chunk = (resampled_real + 1j * resampled_imag).astype(np.complex128)

            # Trim or pad to the exactly calculated target length
            current_len = len(upsampled_chunk)
            if current_len > num_samples_chunk_recon:
                upsampled_chunk = upsampled_chunk[:num_samples_chunk_recon]
            elif current_len < num_samples_chunk_recon:
                pad_length = num_samples_chunk_recon - current_len
                upsampled_chunk = np.pad(upsampled_chunk, (0, pad_length), mode='constant')

            # Maintain RMS consistency (resample_poly is approx power-preserving)
            current_rms = np.sqrt(np.mean(np.abs(upsampled_chunk)**2))
            if current_rms > 1e-12 and rms_in > 1e-12:
                scale_correction = rms_in / current_rms
                upsampled_chunk *= scale_correction

            upsampled_chunks_list.append(upsampled_chunk.copy())

            # Plotting first chunk if requested
            if i == 0 and plot_first:
                print("Plotting first upsampled chunk (polyphase method)...")
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
                plt.xlim(min(time_axis_debug)-0.1, min(max(time_axis_debug), 5.0)) # Adjust xlim
                plt.show(block=False); plt.pause(0.1)

        except Exception as resample_e:
            print(f"Error resampling chunk {i}: {resample_e}. Appending zeros.")
            upsampled_chunks_list.append(np.zeros(num_samples_chunk_recon, dtype=complex))

    return upsampled_chunks_list