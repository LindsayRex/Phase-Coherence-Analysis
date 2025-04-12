import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import h5py
from tqdm import tqdm
import sys
import pywt  # For wavelet transforms

# --- Parameters ---
input_filename = "simulated_chunks_25GHz_400MHzBW_qam16_sdr56MHz.h5"
stitching_window_type = 'hann'
plot_length = 5000
spectrum_ylim = -100
EXPECTED_RMS = 1.39e-02

# --- 1. Load Simulated Chunk Data ---
print(f"Loading data from: {input_filename}")
loaded_chunks = []
loaded_metadata = []
global_attrs = {}
try:
    with h5py.File(input_filename, 'r') as f:
        for key, value in f.attrs.items(): global_attrs[key] = value
        print("--- Global Parameters ---")
        for key, value in global_attrs.items(): print(f"{key}: {value}")
        print("-------------------------")
        actual_chunks = global_attrs.get('actual_num_chunks_saved', 0)
        if actual_chunks == 0: raise ValueError("No chunks.")
        for i in range(actual_chunks):
            group = f[f'chunk_{i:03d}']
            chunk_data = group['iq_data'][:].astype(np.complex128)
            meta = {key: value for key, value in group.attrs.items()}
            loaded_chunks.append(chunk_data)
            loaded_metadata.append(meta)
except Exception as e:
    print(f"Error loading HDF5 file '{input_filename}': {e}"); sys.exit(1)
if not loaded_chunks: print("No chunk data loaded. Exiting."); sys.exit(1)
print(f"\nSuccessfully loaded {len(loaded_chunks)} chunks.")

# Define global constants from loaded attributes
GLBL_SDR_SAMPLE_RATE = global_attrs.get('sdr_sample_rate_hz', None)
GLBL_RECON_SAMPLE_RATE = global_attrs.get('ground_truth_sample_rate_hz', None)
GLBL_OVERLAP_FACTOR = global_attrs.get('overlap_factor', 0.1)
GLBL_TUNING_DELAY = global_attrs.get('tuning_delay_s', 5e-6)
SYMBOL_RATE = global_attrs['total_signal_bandwidth_hz'] / 4  # 16-QAM: 400 MHz / 4 = 100 Msps

# Validate the loaded parameters
if GLBL_SDR_SAMPLE_RATE is None or GLBL_RECON_SAMPLE_RATE is None:
    print("Error: Sample rate information missing.")
    sys.exit(1)
if GLBL_SDR_SAMPLE_RATE <= 0 or GLBL_RECON_SAMPLE_RATE <= 0:
    print("Error: Invalid sample rates.")
    sys.exit(1)

# --- 1b. Correct Initial Amplitude Scaling ---
print("\n--- Correcting Initial Amplitude Scaling ---")
scaled_loaded_chunks = []
print("Chunk | RMS Before | Scaling Factor | RMS After  | Max Abs (After)")
print("------|------------|----------------|------------|----------------")
scaling_successful = True
for i, chunk in enumerate(loaded_chunks):
    if len(chunk) == 0:
        scaled_loaded_chunks.append(chunk)
        print(f"{i:<5d} | --- EMPTY ---    | ---            | --- EMPTY ---    | ---")
        continue
    if not np.all(np.isfinite(chunk)):
        print(f"ERROR: Chunk {i} non-finite BEFORE scaling.")
        scaling_successful = False
        scaled_loaded_chunks.append(chunk)
        continue
    rms_before_scaling = np.sqrt(np.mean(np.abs(chunk)**2))
    max_abs_before = np.max(np.abs(chunk))
    if rms_before_scaling < 1e-12:
        print(f"{i:<5d} | {rms_before_scaling:.4e}       | SKIPPED (Zero) | {rms_before_scaling:.4e}       | {max_abs_before:.4e}")
        scaled_loaded_chunks.append(chunk)
        continue
    scaling_factor = EXPECTED_RMS / rms_before_scaling
    scaled_chunk = (chunk * scaling_factor).astype(np.complex128)
    rms_after_scaling = np.sqrt(np.mean(np.abs(scaled_chunk)**2))
    max_abs_after = np.max(np.abs(scaled_chunk))
    if not np.isclose(rms_after_scaling, EXPECTED_RMS, rtol=1e-3):
        print(f"WARNING: Chunk {i} RMS scaling mismatch ({rms_after_scaling:.4e} vs {EXPECTED_RMS:.4e})")
    print(f"{i:<5d} | {rms_before_scaling:.4e} | {scaling_factor:<14.4f} | {rms_after_scaling:.4e} | {max_abs_after:.4e}")
    scaled_loaded_chunks.append(scaled_chunk)
if not scaling_successful:
    print("\nERROR: Non-finite values detected. Cannot proceed.")
    sys.exit(1)
print("--- Initial Amplitude Scaling Complete ---")

# --- Adaptive Filtering Pre-Alignment ---
print("\n--- Performing Adaptive Filtering Pre-Alignment ---")
def lms_filter(reference, input_signal, mu=0.01, filter_length=32):
    n = len(input_signal)
    w = np.zeros(filter_length, dtype=complex)  # Filter coefficients
    y = np.zeros(n, dtype=complex)  # Output signal
    e = np.zeros(n, dtype=complex)  # Error signal
    for i in range(filter_length, n):
        x = input_signal[i-filter_length:i][::-1]
        y[i] = np.dot(w, x)
        e[i] = reference[i] - y[i]
        w += mu * e[i] * np.conj(x)
    return y, e

chunk_duration_s = len(scaled_loaded_chunks[0]) / GLBL_SDR_SAMPLE_RATE if GLBL_SDR_SAMPLE_RATE > 0 else 0
overlap_samples = int(round(chunk_duration_s * GLBL_OVERLAP_FACTOR * GLBL_SDR_SAMPLE_RATE))
aligned_chunks = [scaled_loaded_chunks[0]]  # First chunk as reference
for i in range(1, len(scaled_loaded_chunks)):
    if len(scaled_loaded_chunks[i]) == 0:
        aligned_chunks.append(scaled_loaded_chunks[i])
        continue
    ref_chunk = aligned_chunks[-1][-overlap_samples:]  # Reference: end of previous chunk
    curr_chunk = scaled_loaded_chunks[i][:overlap_samples]  # Current: start of current chunk
    min_len = min(len(ref_chunk), len(curr_chunk))
    ref_chunk = ref_chunk[:min_len]
    curr_chunk = curr_chunk[:min_len]
    y, e = lms_filter(ref_chunk, curr_chunk)
    phase_correction = np.angle(np.mean(e))
    amplitude_correction = np.mean(np.abs(ref_chunk)) / np.mean(np.abs(curr_chunk)) if np.mean(np.abs(curr_chunk)) > 1e-12 else 1.0
    corrected_chunk = scaled_loaded_chunks[i] * amplitude_correction * np.exp(-1j * phase_correction)
    # Normalize RMS to target
    rms_corrected = np.sqrt(np.mean(np.abs(corrected_chunk)**2))
    if rms_corrected > 1e-12:
        corrected_chunk *= EXPECTED_RMS / rms_corrected
    aligned_chunks.append(corrected_chunk)
print("--- Adaptive Filtering Pre-Alignment Complete ---")

# --- 2. Upsample Chunks (FFT Method) ---
print("\n--- Upsampling chunks (using FFT method) ---")
upsampled_chunks = []
debug_rms_upsample_input = []
debug_rms_upsample_output = []
debug_max_abs_upsample_input = []
debug_max_abs_upsample_output = []
for i, chunk_data in tqdm(enumerate(aligned_chunks), total=len(aligned_chunks), desc="Upsampling"):
    meta = loaded_metadata[i]
    chunk_duration = len(chunk_data)/GLBL_SDR_SAMPLE_RATE if GLBL_SDR_SAMPLE_RATE > 0 else 0
    num_samples_chunk_recon = int(round(chunk_duration * GLBL_RECON_SAMPLE_RATE))
    if len(chunk_data) < 2:
        upsampled_chunks.append(np.zeros(num_samples_chunk_recon, dtype=complex))
        debug_rms_upsample_input.append(0.0)
        debug_rms_upsample_output.append(0.0)
        debug_max_abs_upsample_input.append(0.0)
        debug_max_abs_upsample_output.append(0.0)
        continue
    try:
        rms_in = np.sqrt(np.mean(np.abs(chunk_data)**2))
        max_abs_in = np.max(np.abs(chunk_data))
        debug_rms_upsample_input.append(rms_in)
        debug_max_abs_upsample_input.append(max_abs_in)
        if rms_in < 1e-12:
            upsampled_chunks.append(np.zeros(num_samples_chunk_recon, dtype=complex))
            debug_rms_upsample_output.append(0.0)
            debug_max_abs_upsample_output.append(0.0)
            continue
        n_orig = len(chunk_data)
        n_new = num_samples_chunk_recon
        fft_orig = np.fft.fft(chunk_data)
        fft_shifted = np.fft.fftshift(fft_orig)
        n_half = (n_orig + 1) // 2
        fft_padded = np.zeros(n_new, dtype=complex)
        start_idx = (n_new - n_orig) // 2
        fft_padded[start_idx:start_idx+n_orig] = fft_shifted
        fft_padded = np.fft.ifftshift(fft_padded)
        upsampled_chunk = np.fft.ifft(fft_padded) * (n_new / n_orig)
        upsampled_chunk = upsampled_chunk.astype(np.complex128)
        rms_out = np.sqrt(np.mean(np.abs(upsampled_chunk)**2))
        max_abs_out = np.max(np.abs(upsampled_chunk))
        debug_rms_upsample_output.append(rms_out)
        debug_max_abs_upsample_output.append(max_abs_out)
        upsampled_chunks.append(upsampled_chunk.copy())
        if i == 0:
            print("Plotting first upsampled chunk (FFT method)...")
            plt.figure(figsize=(12,4))
            time_axis_debug = np.arange(len(upsampled_chunk))/GLBL_RECON_SAMPLE_RATE*1e6
            plt.plot(time_axis_debug, upsampled_chunk.real, label='Real')
            plt.plot(time_axis_debug, upsampled_chunk.imag, label='Imag', alpha=0.7)
            plt.title('First Upsampled Chunk')
            plt.xlabel('Time (µs)')
            plt.ylabel('Amp')
            plt.legend()
            plt.grid(True)
            plot_limit_samples = min(len(upsampled_chunk), int(5*GLBL_RECON_SAMPLE_RATE*1e-6))
            x_limit = time_axis_debug[plot_limit_samples] if plot_limit_samples > 0 and plot_limit_samples < len(time_axis_debug) else 5
            plt.xlim(-0.3, x_limit)
            plt.ylim(-1.5, 1.5)
            plt.show()
            print("Plotting Spectra Comparison (After Upsampling)...")
            plt.figure(figsize=(12, 6))
            n = len(chunk_data)
            f = np.fft.fftshift(np.fft.fftfreq(n, d=1/GLBL_SDR_SAMPLE_RATE))
            s = np.fft.fftshift(np.fft.fft(chunk_data))
            db = 20*np.log10(np.abs(s)+1e-12)
            db -= np.max(db)
            plt.plot(f/1e6, db, label=f'BEFORE (Fs={GLBL_SDR_SAMPLE_RATE/1e6:.1f}MHz)', alpha=0.8)
            n = len(upsampled_chunk)
            f = np.fft.fftshift(np.fft.fftfreq(n, d=1/GLBL_RECON_SAMPLE_RATE))
            s = np.fft.fftshift(np.fft.fft(upsampled_chunk))
            db = 20*np.log10(np.abs(s)+1e-12)
            db -= np.max(db)
            plt.plot(f/1e6, db, label=f'AFTER (Fs={GLBL_RECON_SAMPLE_RATE/1e6:.1f}MHz)', ls='--', alpha=0.8)
            plt.title('Spectra Comparison Upsampling')
            plt.xlabel('Freq (MHz rel Chunk Center)')
            plt.ylabel('Mag (dB)')
            plt.ylim(-120)
            plt.legend()
            plt.grid(True)
            plt.show()
    except Exception as resample_e:
        print(f"Error resampling chunk {i}: {resample_e}. Appending zeros.")
        upsampled_chunks.append(np.zeros(num_samples_chunk_recon, dtype=complex))
        debug_rms_upsample_input.append(0.0)
        debug_rms_upsample_output.append(0.0)
        debug_max_abs_upsample_input.append(0.0)
        debug_max_abs_upsample_output.append(0.0)
print("\n--- RMS and Max Abs Before/After Upsampling (FFT Method) ---")
print("Chunk | RMS Before | Max Abs Before | RMS After  | Max Abs After")
print("------|------------|----------------|------------|--------------")
min_len_rms = min(len(debug_rms_upsample_input), len(debug_rms_upsample_output))
for i in range(min_len_rms):
    print(f"{i:<5d} | {debug_rms_upsample_input[i]:.4e} | {debug_max_abs_upsample_input[i]:.4e}   | {debug_rms_upsample_output[i]:.4e} | {debug_max_abs_upsample_output[i]:.4e}")

# --- 3. Adaptive Wavelet Design and Phase-Locking ---
print("\n--- Performing Adaptive Wavelet Design and Phase-Locking ---")
# Step 1: Approximate an adaptive wavelet for 16-QAM
wavelet_base = 'db4'  # Daubechies wavelet as a starting point
level = 4  # Decomposition level

# Ensure chunk length is a multiple of 2^level
chunk_length = len(upsampled_chunks[0])
required_length = chunk_length
while required_length % (2**level) != 0:
    required_length += 1
adjusted_chunks = []
for chunk in upsampled_chunks:
    if len(chunk) == 0:
        adjusted_chunks.append(chunk)
        continue
    padded_chunk = np.pad(chunk, (0, required_length - chunk_length), mode='constant')
    adjusted_chunks.append(padded_chunk)

# Step 2: Decompose each chunk using DWT with the selected wavelet
overlap_samples_recon = int(round(chunk_duration_s * GLBL_OVERLAP_FACTOR * GLBL_RECON_SAMPLE_RATE))
overlap_samples_downsampled = overlap_samples_recon // (2**level)  # Adjust for downsampling
if overlap_samples_downsampled < 1:
    overlap_samples_downsampled = 1
    print("Warning: Downsampled overlap samples too small. Setting to 1.")

dwt_coeffs = []
for chunk in adjusted_chunks:
    if len(chunk) == 0:
        dwt_coeffs.append(None)
        continue
    coeffs = pywt.wavedec(chunk, wavelet_base, level=level, mode='per')
    dwt_coeffs.append(coeffs)

# Step 3: Phase-Locking
phase_corrected_dwt_coeffs = [dwt_coeffs[0]] if dwt_coeffs[0] is not None else [None]
for i in range(1, len(dwt_coeffs)):
    if dwt_coeffs[i] is None or dwt_coeffs[i-1] is None:
        phase_corrected_dwt_coeffs.append(dwt_coeffs[i])
        continue
    # Compute phase shift in the overlap region using approximation coefficients
    cA1 = dwt_coeffs[i-1][0][-overlap_samples_downsampled:]  # Approximation coefficients of previous chunk
    cA2 = dwt_coeffs[i][0][:overlap_samples_downsampled]     # Approximation coefficients of current chunk
    if len(cA1) != len(cA2):
        min_len = min(len(cA1), len(cA2))
        cA1 = cA1[:min_len]
        cA2 = cA2[:min_len]
    if len(cA1) == 0:
        phase_corrected_dwt_coeffs.append(dwt_coeffs[i])
        continue
    # Compute phase difference: \Delta\phi = (1/N) \sum \angle(c_{j,k}^1 / (c_{j,k}^2)*)
    phase_diffs = np.angle(cA1 / np.conj(cA2))
    phase_diffs = phase_diffs[np.isfinite(phase_diffs)]  # Remove non-finite values
    if len(phase_diffs) == 0:
        delta_phi = 0.0
    else:
        delta_phi = np.mean(phase_diffs)
    # Adjust only the approximation coefficients (cA), leave detail coefficients unchanged
    corrected_coeffs = [dwt_coeffs[i][0] * np.exp(-1j * delta_phi)]  # cA
    corrected_coeffs.extend(dwt_coeffs[i][1:])  # cD_n, ..., cD_1 (unchanged)
    phase_corrected_dwt_coeffs.append(corrected_coeffs)

# Step 4: Reconstruct chunks using inverse DWT
phase_corrected_chunks = []
for i, coeffs in enumerate(phase_corrected_dwt_coeffs):
    if coeffs is None:
        chunk_duration = len(upsampled_chunks[i])/GLBL_RECON_SAMPLE_RATE if GLBL_RECON_SAMPLE_RATE > 0 else 0
        num_samples_chunk_recon = int(round(chunk_duration * GLBL_RECON_SAMPLE_RATE))
        phase_corrected_chunks.append(np.zeros(num_samples_chunk_recon, dtype=complex))
        continue
    reconstructed_chunk = pywt.waverec(coeffs, wavelet_base, mode='per')
    # Trim back to original length
    original_length = len(upsampled_chunks[i])
    reconstructed_chunk = reconstructed_chunk[:original_length]
    phase_corrected_chunks.append(reconstructed_chunk)

print("--- Adaptive Wavelet Design and Phase-Locking Complete ---")

# --- 4. Stitching (Overlap-Add) ---
print("\n--- Performing Wavelet Processing AT RECONSTRUCTION RATE ---")
time_advance_per_chunk = chunk_duration_s * (1.0 - GLBL_OVERLAP_FACTOR)
total_duration_recon = chunk_duration_s + (len(loaded_chunks) - 1) * (time_advance_per_chunk + GLBL_TUNING_DELAY)
num_samples_recon = int(round(total_duration_recon * GLBL_RECON_SAMPLE_RATE))

reconstructed_signal = np.zeros(num_samples_recon, dtype=complex)
sum_of_windows = np.zeros(num_samples_recon, dtype=float)

print(f"Reconstruction target buffer: {num_samples_recon} samples @ {GLBL_RECON_SAMPLE_RATE/1e6:.2f} MHz (Est. Duration: {total_duration_recon*1e6:.1f} us)")

current_recon_time_start = 0.0
plt.figure(figsize=(14, 8))
num_subplot_rows = int(np.ceil(len(phase_corrected_chunks)/2))
debug_plot_len = int(chunk_duration_s * GLBL_RECON_SAMPLE_RATE * 1.2)

for i, up_chunk in tqdm(enumerate(phase_corrected_chunks), total=len(phase_corrected_chunks), desc="Wavelet Recon"):
    if i >= len(loaded_metadata):
        print(f"Warning: Index {i} exceeds loaded_metadata length ({len(loaded_metadata)}). Skipping chunk.")
        if i < len(loaded_chunks) - 1: current_recon_time_start += time_advance_per_chunk + GLBL_TUNING_DELAY
        continue
    meta = loaded_metadata[i]
    if len(up_chunk) == 0:
        if i < len(loaded_chunks) - 1: current_recon_time_start += time_advance_per_chunk + GLBL_TUNING_DELAY
        continue

    rms_in_stitching = np.sqrt(np.mean(np.abs(up_chunk)**2))
    print(f"Chunk {i} RMS: Before= {rms_in_stitching:.4e}")
    max_abs_chunk = np.max(np.abs(up_chunk))
    print(f"Chunk {i} Max Abs: {max_abs_chunk:.4e}")

    start_idx_recon = int(round(current_recon_time_start * GLBL_RECON_SAMPLE_RATE))
    num_samples_in_chunk = len(up_chunk)
    end_idx_recon = min(start_idx_recon + num_samples_in_chunk, num_samples_recon)
    actual_len = end_idx_recon - start_idx_recon
    if actual_len <= 0:
        if i < len(loaded_chunks) - 1: current_recon_time_start += time_advance_per_chunk + GLBL_TUNING_DELAY
        continue

    if actual_len < 2:
        window = np.ones(actual_len)
    else:
        window = sig.get_window(stitching_window_type, actual_len)
        window *= np.sqrt(actual_len / np.sum(window**2))  # Normalize for unity gain

    try:
        segment_to_add = up_chunk[:actual_len] * window
        if not np.all(np.isfinite(segment_to_add)):
            print(f"*** WARNING: Non-finite segment_to_add chunk {i} ***")
            segment_to_add[~np.isfinite(segment_to_add)] = 0
        reconstructed_signal[start_idx_recon:end_idx_recon] += segment_to_add
        if not np.all(np.isfinite(reconstructed_signal[start_idx_recon:end_idx_recon])):
            print(f"*** WARNING: Non-finite recon after adding chunk {i} ***")
        sum_of_windows[start_idx_recon:end_idx_recon] += window
    except Exception as add_e:
        print(f"Error during overlap-add chunk {i}: {add_e}")
        continue

    if i < num_subplot_rows * 2:
        ax = plt.subplot(num_subplot_rows, 2, i + 1)
        plot_end_idx = min(end_idx_recon + int(0.1*num_samples_in_chunk), num_samples_recon)
        time_axis = np.arange(start_idx_recon, plot_end_idx) / GLBL_RECON_SAMPLE_RATE * 1e6
        if len(time_axis) > 0:
            plot_data = reconstructed_signal[start_idx_recon:plot_end_idx]
            max_val_plot = np.nanmax(np.abs(plot_data)) if len(plot_data) > 0 else 1.0
            if max_val_plot < 1e-12 or not np.isfinite(max_val_plot):
                max_val_plot = 1.0
            ax.plot(time_axis, plot_data.real, label=f'Real (After Chunk {i})')
            ax.plot(time_axis, plot_data.imag, label=f'Imag (After Chunk {i})', alpha=0.7)
            ax.set_title(f'Recon State After Chunk {i}', fontsize='small')
            ax.grid(True)
            ax.set_ylim(-max_val_plot*1.5, max_val_plot*1.5)
        else:
            ax.set_title(f'Chunk {i} - No data', fontsize='small')

    if i < len(loaded_chunks) - 1:
        current_recon_time_start += time_advance_per_chunk + GLBL_TUNING_DELAY

plt.suptitle("Progressive Reconstruction Signal (Overlap-Add)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
print("\nWavelet reconstruction and stitching loop complete.")

# Debug: Plot sum_of_windows
plt.figure(figsize=(12, 4))
time_axis = np.arange(len(sum_of_windows)) / GLBL_RECON_SAMPLE_RATE * 1e6
plt.plot(time_axis, sum_of_windows, label='Sum of Windows')
plt.axhline(y=1e-6, color='r', linestyle='--', label='Reliable Threshold (1e-6)')
plt.title('Sum of Windows Across Reconstructed Signal')
plt.xlabel('Time (µs)')
plt.ylabel('Window Sum')
plt.legend()
plt.grid(True)
plt.show()

# --- Normalization Step ---
print("\n--- Normalizing reconstructed signal ---")
rms_before_norm = np.sqrt(np.mean(np.abs(reconstructed_signal)**2))
max_abs_before_norm = np.nanmax(np.abs(reconstructed_signal)) if len(reconstructed_signal) > 0 else 0
print("Signal Stats BEFORE Normalization:")
print(f"  RMS: {rms_before_norm:.4e}")
print(f"  Max Abs: {max_abs_before_norm:.4e}")
reliable_threshold = 1e-6
reliable_indices = np.where(sum_of_windows >= reliable_threshold)[0]
if len(reliable_indices) > 0:
    fallback_sum = np.median(sum_of_windows[reliable_indices])
else:
    fallback_sum = 1.0
print(f"Using median sum_of_windows ({fallback_sum:.4f}) as fallback divisor.")
sum_of_windows_modified = sum_of_windows.copy()
sum_of_windows_modified[sum_of_windows < reliable_threshold] = fallback_sum
reconstructed_signal_normalized = reconstructed_signal / sum_of_windows_modified
rms_after_norm = np.sqrt(np.mean(np.abs(reconstructed_signal_normalized)**2))
max_abs_after_norm = np.nanmax(np.abs(reconstructed_signal_normalized)) if len(reconstructed_signal_normalized) > 0 else 0
print("\nSignal Stats AFTER Normalization:")
print(f"  RMS: {rms_after_norm:.4e}")
print(f"  Max Abs: {max_abs_after_norm:.4e}")
print(f"  Ratio to Initial Target RMS ({EXPECTED_RMS:.4e}): {rms_after_norm/EXPECTED_RMS:.4f}")
reconstructed_signal = reconstructed_signal_normalized
print("\nNormalization complete.")

# --- 5. Evaluation & Visualization ---
print("\n--- Evaluating Reconstruction ---")
print("Regenerating ground truth baseband for comparison...")
gt_duration = total_duration_recon
num_samples_gt_compare = int(round(gt_duration * GLBL_RECON_SAMPLE_RATE))
t_gt_compare = np.linspace(0, gt_duration, num_samples_gt_compare, endpoint=False)
gt_baseband_compare = np.zeros(num_samples_gt_compare, dtype=complex)
mod = global_attrs.get('modulation', 'qam16')
bw_gt = global_attrs['total_signal_bandwidth_hz']
if mod.lower() == 'qam16':
    symbol_rate_gt = bw_gt / 4
    print(f"Using GT Symbol Rate = {symbol_rate_gt/1e6:.2f} Msps")
    num_symbols_gt = int(np.ceil(gt_duration * symbol_rate_gt))
    symbols = (np.random.choice([-3,-1,1,3], size=num_symbols_gt) + 1j*np.random.choice([-3,-1,1,3], size=num_symbols_gt))/np.sqrt(10)
    samples_per_symbol_gt = max(1, int(round(GLBL_RECON_SAMPLE_RATE/symbol_rate_gt)))
    baseband_symbols = np.repeat(symbols, samples_per_symbol_gt)
    len_to_take = min(len(baseband_symbols), num_samples_gt_compare)
    gt_baseband_compare[:len_to_take] = baseband_symbols[:len_to_take]
else:
    print(f"Warning: GT regen not implemented for {mod}")

# Scale GT to match EXPECTED_RMS
gt_rms_before = np.sqrt(np.mean(np.abs(gt_baseband_compare)**2))
if gt_rms_before > 1e-20:
    gt_scale_factor = EXPECTED_RMS / gt_rms_before
    gt_baseband_compare *= gt_scale_factor
    gt_rms_after = np.sqrt(np.mean(np.abs(gt_baseband_compare)**2))
    print(f"Scaled GT baseband. Target RMS: {EXPECTED_RMS:.4e}, Actual RMS: {gt_rms_after:.4e}")
else:
    print("Warning: GT baseband power too low. Skipping scaling.")

# Evaluation
reliable_threshold = 1e-6
reliable_indices = np.where(sum_of_windows >= reliable_threshold)[0]
if len(reliable_indices) < 2:
    mse = np.inf
    nmse = np.inf
    evm = np.inf
    reconstructed_signal_aligned = reconstructed_signal
else:
    print(f"Evaluating metrics using {len(reliable_indices)} reliable samples.")
    gt_reliable = gt_baseband_compare[reliable_indices]
    recon_reliable = reconstructed_signal[reliable_indices]
    mean_power_gt_reliable = np.mean(np.abs(gt_reliable)**2)
    gt_reliable_norm = gt_reliable / np.sqrt(mean_power_gt_reliable) if mean_power_gt_reliable > 1e-20 else gt_reliable
    mean_power_recon_reliable = np.mean(np.abs(recon_reliable)**2)
    if mean_power_recon_reliable > 1e-20 and np.isfinite(mean_power_recon_reliable):
        recon_reliable_norm = recon_reliable / np.sqrt(mean_power_recon_reliable)
    else:
        print("Warning: Reliable Recon power invalid.")
        recon_reliable_norm = recon_reliable
    if not np.all(np.isfinite(recon_reliable_norm)):
        mse = np.inf
        nmse = np.inf
        evm = np.inf
    else:
        len_gt_rel = len(gt_reliable_norm)
        len_rec_rel = len(recon_reliable_norm)
        if len_gt_rel != len_rec_rel:
            min_rel_len = min(len_gt_rel, len_rec_rel)
            gt_reliable_norm = gt_reliable_norm[:min_rel_len]
            recon_reliable_norm = recon_reliable_norm[:min_rel_len]
        mse = np.mean(np.abs(gt_reliable_norm - recon_reliable_norm)**2)
        nmse = mse / np.mean(np.abs(gt_reliable_norm)**2) if np.mean(np.abs(gt_reliable_norm)**2) > 1e-20 else np.inf
        evm = np.sqrt(mse) * 100  # EVM in percent
    reconstructed_signal_aligned = np.zeros_like(reconstructed_signal)
    if mean_power_recon_reliable > 1e-20 and np.isfinite(mean_power_recon_reliable) and mean_power_gt_reliable > 1e-20:
        power_scale_factor = np.sqrt(mean_power_gt_reliable) / np.sqrt(mean_power_recon_reliable)
        reconstructed_signal_aligned[reliable_indices] = recon_reliable * power_scale_factor
        print(f"Applied plotting alignment scale factor: {power_scale_factor:.4f}")
    else:
        reconstructed_signal_aligned[reliable_indices] = recon_reliable

print("\nEvaluation Metrics:")
print(f"  MSE : {mse:.4e}")
print(f"  NMSE: {nmse:.4e} ({10*np.log10(nmse):.2f} dB)")
print(f"  EVM : {evm:.2f}%")

# --- Plotting ---
print("\n--- Generating Matplotlib Plots ---")
plt.style.use('seaborn-v0_8-darkgrid')
fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
mean_power_gt_plot = np.mean(np.abs(gt_baseband_compare)**2)
gt_plot_norm = gt_baseband_compare / np.sqrt(mean_power_gt_plot) if mean_power_gt_plot > 1e-20 else gt_baseband_compare
plot_samples = min(plot_length, len(t_gt_compare), len(gt_plot_norm))
time_axis_plot = t_gt_compare[:plot_samples] * 1e6
axs[0].plot(time_axis_plot, gt_plot_norm[:plot_samples].real, label='GT (Real)')
axs[0].plot(time_axis_plot, gt_plot_norm[:plot_samples].imag, label='GT (Imag)', alpha=0.7)
axs[0].set_title('Ground Truth (Normalized)')
axs[0].set_ylabel('Amp')
axs[0].legend(fontsize='small')
axs[0].grid(True)
plot_samples_recon = min(plot_length, len(reconstructed_signal_aligned))
plot_data_recon = reconstructed_signal_aligned[:plot_samples_recon]
plot_data_recon[~np.isfinite(plot_data_recon)] = 0
axs[1].plot(time_axis_plot[:plot_samples_recon], plot_data_recon.real, label='Recon (Real)')
axs[1].plot(time_axis_plot[:plot_samples_recon], plot_data_recon.imag, label='Recon (Imag)', alpha=0.7)
axs[1].set_title(f'Reconstructed (EVM: {evm:.2f}%)')
axs[1].set_ylabel('Amp')
axs[1].legend(fontsize='small')
axs[1].grid(True)
plot_samples_error = min(plot_samples, plot_samples_recon)
error_signal = gt_plot_norm[:plot_samples_error] - plot_data_recon[:plot_samples_error]
axs[2].plot(time_axis_plot[:plot_samples_error], error_signal.real, label='Error (Real)')
axs[2].plot(time_axis_plot[:plot_samples_error], error_signal.imag, label='Error (Imag)', alpha=0.7)
axs[2].set_title('Error')
axs[2].set_xlabel('Time (µs)')
axs[2].set_ylabel('Amp')
axs[2].legend(fontsize='small')
axs[2].grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 7))
if len(reconstructed_signal_aligned) > 1 and np.all(np.isfinite(reconstructed_signal_aligned)):
    f_recon = np.fft.fftshift(np.fft.fftfreq(len(reconstructed_signal_aligned), d=1/GLBL_RECON_SAMPLE_RATE))
    spec_recon = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(reconstructed_signal_aligned)))+1e-12)
    spec_recon -= np.nanmax(spec_recon)
    plt.plot(f_recon/1e6, spec_recon, label='Recon Spec', ls='--', alpha=0.8)
else:
    print("Skipping recon spectrum plot.")
if len(gt_plot_norm) > 1:
    f_gt = np.fft.fftshift(np.fft.fftfreq(len(gt_plot_norm), d=1/GLBL_RECON_SAMPLE_RATE))
    spec_gt = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(gt_plot_norm)))+1e-12)
    spec_gt -= np.nanmax(spec_gt)
    plt.plot(f_gt/1e6, spec_gt, label='GT Spec', alpha=0.8)
else:
    print("Skipping GT spectrum plot.")
plt.title('Spectra Comparison')
plt.xlabel('Freq (MHz)')
plt.ylabel('Mag (dB)')
plt.ylim(spectrum_ylim)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nScript finished.")