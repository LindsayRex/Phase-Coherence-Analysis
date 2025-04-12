import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import h5py
from tqdm import tqdm
import pywt  # For Wavelets
import tensorly as tl  # For Tensor Decomposition (CP)
from tensorly.decomposition import parafac

# --- Tuning Parameters ---
# These parameters control the behavior of the reconstruction process.
# Adjust them to optimize performance based on your signal characteristics.

# Input File
input_filename = "simulated_chunks_25GHz_400MHzBW_qam16_sdr56MHz.h5"
# Description: Path to the HDF5 file containing simulated chunk data.
# Impact: Must match the output file from your data generation script.

# Adaptive Filter Parameters
filter_length = 32
# Description: Number of taps in the transversal filter for LMS adaptation.
# Impact: Larger values capture longer delays/phase shifts but increase computation. Typical range: 16-64.
mu = 0.01
# Description: Step size for LMS adaptation in adaptive filtering.
# Impact: Controls convergence speed vs. stability. Smaller values (e.g., 0.001) are more stable but slower; larger values (e.g., 0.1) converge faster but may diverge.

# CP Decomposition Parameters
cp_rank = 10
# Description: Rank of the CP decomposition (number of components to extract).
# Impact: Should match the expected number of signal components in each chunk. Too high: overfitting/noise; too low: misses components. Typical range: 3-10.
num_lags = 5
# Description: Number of time lags used to build the correlation tensor for CP decomposition.
# Impact: More lags improve frequency resolution but require more samples and computation. Typical range: 3-10.
delay_samples = 1
# Description: Number of samples to delay the chunk for the delayed path in CP decomposition.
# Impact: Larger delays improve frequency resolution (via phase shift) but reduce usable samples. Typical range: 1-10.
cp_iter_max = 200
# Description: Maximum iterations for CP decomposition convergence.
# Impact: Higher values ensure convergence but increase computation time. Typical range: 100-500.
cp_tolerance = 1e-6
# Description: Convergence tolerance for CP decomposition.
# Impact: Smaller values improve accuracy but may increase iterations. Typical range: 1e-6 to 1e-8.

# Wavelet Parameters
wavelet_name = 'db4'
# Description: Type of wavelet used for Wavelet Packet Decomposition (WPD).
# Impact: Affects time-frequency resolution and boundary effects. 'db4' (Daubechies 4) is a good default; alternatives: 'sym4', 'coif2'.
wpd_level = 4 # KEEP THIS REASONABLE - high levels + high sample rate = HUGE computation
# Description: Decomposition level for WPD.
# Impact: Higher levels provide finer frequency resolution but increase computation and may introduce artifacts. Typical range: 3-6.
stitching_window_type = 'hann'
# Description: Window type for overlap-add stitching of reconstructed chunks.
# Impact: Affects smoothness at chunk boundaries. 'hann' is smooth; alternatives: 'hamming', 'blackman'.

# Visualization Parameters
plot_length = 5000
# Description: Number of samples to plot in time-domain visualizations.
# Impact: Larger values show more of the signal but may clutter the plot. Adjust based on signal duration.
spectrum_ylim = -100
# Description: Lower limit (in dB) for the spectrum plot y-axis.
# Impact: Adjusts visibility of low-power spectral components. Typical range: -80 to -120.

# --- 1. Load Simulated Chunk Data ---
print(f"Loading data from: {input_filename}")
loaded_chunks = []
loaded_metadata = []
global_attrs = {}

try:
    with h5py.File(input_filename, 'r') as f:
        for key, value in f.attrs.items():
            global_attrs[key] = value
        print("--- Global Parameters ---")
        for key, value in global_attrs.items():
            print(f"{key}: {value}")
        print("-------------------------")

        actual_chunks = global_attrs.get('actual_num_chunks_saved', 0)
        if actual_chunks == 0:
            raise ValueError("No chunks found in HDF5 file.")

        for i in range(actual_chunks):
            chunk_name = f'chunk_{i:03d}'
            if chunk_name in f:
                group = f[chunk_name]
                chunk_data = group['iq_data'][:]
                meta = {key: value for key, value in group.attrs.items()}
                loaded_chunks.append(chunk_data)
                loaded_metadata.append(meta)
            else:
                print(f"Warning: Chunk {chunk_name} not found.")
except Exception as e:
    print(f"Error loading HDF5 file '{input_filename}': {e}")
    exit()

if not loaded_chunks:
    print("No chunk data loaded. Exiting.")
    exit()

print(f"\nSuccessfully loaded {len(loaded_chunks)} chunks.")
fs_sdr = global_attrs['sdr_sample_rate_hz'] # Get SDR sample rate from global attrs

# --- 2. Adaptive Filtering Pre-Alignment ---
# (Skipping implementation details for brevity - assume it runs as before)
print("\n--- Performing Adaptive Filtering Pre-Alignment (as before) ---")
aligned_chunks = [loaded_chunks[0].copy()] # Start with a copy of the first chunk

for i in tqdm(range(1, len(loaded_chunks)), desc="Aligning chunks"):
    # --- (Your AF code using LMS and convolution as before) ---
    prev_chunk = loaded_chunks[i - 1]
    curr_chunk = loaded_chunks[i]
    meta_prev = loaded_metadata[i - 1]
    meta_curr = loaded_metadata[i]

    overlap_factor = global_attrs['overlap_factor']
    chunk_samples = len(curr_chunk)
    overlap_samples = int(chunk_samples * overlap_factor)
    if overlap_samples < filter_length: # Ensure enough samples for LMS
        print(f"Warning: Chunk {i} overlap too small ({overlap_samples}) for filter length ({filter_length}). Skipping AF.")
        aligned_chunks.append(curr_chunk.copy()) # Use original chunk if AF skipped
        continue

    # Extract overlapping regions
    ref_signal = prev_chunk[-overlap_samples:]
    input_signal = curr_chunk[:overlap_samples]

    # Coarse delay estimation via cross-correlation
    corr = sig.correlate(ref_signal, input_signal, mode='full')
    delay_est = np.argmax(np.abs(corr)) - (len(input_signal) - 1)
    input_signal_shifted = np.roll(input_signal, delay_est)
    if delay_est > 0:
        input_signal_shifted[:delay_est] = 0
    elif delay_est < 0:
        input_signal_shifted[delay_est:] = 0

    # LMS adaptation
    weights = np.zeros(filter_length, dtype=complex)
    y = np.zeros(len(input_signal_shifted), dtype=complex)
    try: # Add try-except for potential numerical issues in LMS
        for n in range(filter_length, len(input_signal_shifted)):
            x_n = input_signal_shifted[n - filter_length:n][::-1]
            y[n] = np.dot(weights, x_n)
            error = ref_signal[n] - y[n]
            weights += mu * error * np.conj(x_n)
            if not np.all(np.isfinite(weights)):
                 print(f"Warning: Non-finite weights encountered in LMS for chunk {i}. Resetting weights.")
                 weights = np.zeros(filter_length, dtype=complex) # Reset
                 break # Stop LMS for this chunk
    except Exception as lms_e:
        print(f"Error during LMS for chunk {i}: {lms_e}")
        weights = np.zeros(filter_length, dtype=complex) # Use zero weights if error

    # Apply filter to entire current chunk
    if np.all(np.isfinite(weights)): # Only apply if weights are valid
        # Use 'same' mode to avoid length change, apply carefully
        aligned_chunk_conv = sig.convolve(curr_chunk, weights, mode='same')
        aligned_chunks.append(aligned_chunk_conv)
        # Optional: Add print statement for max weight magnitude here if needed
        # print(f"Chunk {i}: Delay Est={delay_est} samples, Max weight mag={np.max(np.abs(weights)):.4f}")
    else:
        print(f"Warning: Using original chunk {i} due to LMS issues.")
        aligned_chunks.append(curr_chunk.copy())


# --- 3. Sub-Nyquist CP Decomposition (Real Implementation) ---
# (Skipping implementation details for brevity - assume it runs as before)
print("\n--- Performing Real CP Decomposition (as before) ---")
estimated_chunk_phases = []
estimated_chunk_dominant_freqs = []

for i, chunk_data in enumerate(aligned_chunks): # Use aligned_chunks now
    meta = loaded_metadata[i]
    true_applied_phase = meta['applied_phase_offset_rad']

    # Simulate delayed path
    chunk_delayed = np.roll(chunk_data, delay_samples)
    chunk_delayed[:delay_samples] = 0

    # Build correlation tensor
    max_len = len(chunk_data) - num_lags - delay_samples
    if max_len < cp_rank * 2: # Need more samples than rank
        print(f"  Skipping CP on chunk {i}: Not enough samples ({max_len}) after lag/delay.")
        estimated_chunk_phases.append(true_applied_phase)
        estimated_chunk_dominant_freqs.append(0.0)
        continue

    tensor_data = np.zeros((max_len, 2, num_lags), dtype=complex)
    for l in range(num_lags):
        tensor_data[:, 0, l] = chunk_data[l:max_len+l] * np.conj(chunk_data[:max_len])
        tensor_data[:, 1, l] = chunk_delayed[l:max_len+l] * np.conj(chunk_delayed[:max_len])

    # Perform CP Decomposition
    try:
        tl.set_backend('numpy') # Ensure numpy backend is used
        weights, factors = parafac(tensor_data, rank=cp_rank, init='svd', tol=cp_tolerance, n_iter_max=cp_iter_max, verbose=0)
        dominant_comp_idx = np.argmax(weights)
        lag_factor = factors[2][:, dominant_comp_idx]

        # Robust phase slope estimation
        valid_indices = np.where(np.abs(lag_factor) > 1e-6)[0] # Avoid using near-zero factors
        if len(valid_indices) > 1:
            angles = np.angle(lag_factor[valid_indices[1:]] / lag_factor[valid_indices[:-1]])
            dominant_freq_rad_per_lag = np.median(angles) # Median is more robust to outliers
            dominant_freq_hz = dominant_freq_rad_per_lag * sdr_sample_rate / (2 * np.pi)
            phase_est = np.angle(lag_factor[valid_indices[0]]) # Phase of first valid factor
        else:
            print(f"  Warning: Could not reliably estimate phase slope for chunk {i}. Using fallback.")
            dominant_freq_hz = 0.0
            phase_est = true_applied_phase # Fallback

        # --- Debugging Output ---
        # print(f"Chunk {i}: CP Est Freq Offset={dominant_freq_hz/1e3:.1f} kHz, Phase={np.rad2deg(phase_est):.1f} deg")
        # print(f"         True Phase={np.rad2deg(true_applied_phase):.1f} deg")
        # print(f"         Weight Magnitudes: {[abs(w) for w in weights]}")
        # --- End Debugging ---

        estimated_chunk_phases.append(phase_est)
        estimated_chunk_dominant_freqs.append(dominant_freq_hz)

    except Exception as e:
        print(f"  CP decomposition failed for chunk {i}: {e}")
        estimated_chunk_phases.append(true_applied_phase)
        estimated_chunk_dominant_freqs.append(0.0)

print("\nCP decomposition complete.")


# --- 4. Upsample Aligned Chunks FIRST ---
print("\n--- Upsampling chunks BEFORE Wavelet processing ---")
fs_recon = global_attrs['ground_truth_sample_rate_hz']
upsampled_chunks = []

for i, chunk_data in tqdm(enumerate(aligned_chunks), total=len(aligned_chunks), desc="Upsampling"):
    meta = loaded_metadata[i]
    chunk_duration = len(chunk_data) / fs_sdr
    num_samples_chunk_recon = int(round(chunk_duration * fs_recon))

    if len(chunk_data) < 2:
        print(f"Warning: Chunk {i} too short ({len(chunk_data)}) for resampling. Appending zeros.")
        upsampled_chunks.append(np.zeros(num_samples_chunk_recon, dtype=complex))
        continue

    # Use sig.resample (FFT based) instead of resample_poly
    try:
        # Add zero-padding before resampling to reduce edge effects
        padding_len = int(len(chunk_data) * 0.1) # Add 10% padding
        chunk_padded = np.pad(chunk_data, (padding_len, padding_len), mode='constant')

        resampled_padded = sig.resample(chunk_padded, len(chunk_padded) * int(round(fs_recon / fs_sdr)))

        # Calculate expected length after padding and resampling
        target_len_padded_resampled = int(round((len(chunk_data) + 2*padding_len) * (fs_recon / fs_sdr)))
        # Calculate number of samples corresponding to original chunk duration
        target_len_original_resampled = int(round(len(chunk_data) * (fs_recon / fs_sdr)))
        # Calculate number of samples corresponding to padding
        target_len_padding_resampled = int(round(padding_len * (fs_recon / fs_sdr)))

        # Extract the central part corresponding to the original signal duration
        if len(resampled_padded) > target_len_padding_resampled * 2:
            upsampled_chunk = resampled_padded[target_len_padding_resampled:-target_len_padding_resampled]
            # Trim/pad final result to exact expected length
            if len(upsampled_chunk) > num_samples_chunk_recon:
                upsampled_chunk = upsampled_chunk[:num_samples_chunk_recon]
            elif len(upsampled_chunk) < num_samples_chunk_recon:
                upsampled_chunk = np.pad(upsampled_chunk, (0, num_samples_chunk_recon - len(upsampled_chunk)))
            upsampled_chunks.append(upsampled_chunk)
        else:
             print(f"Warning: Resampled chunk {i} length issue after padding removal. Appending zeros.")
             upsampled_chunks.append(np.zeros(num_samples_chunk_recon, dtype=complex))


    except Exception as resample_e:
        print(f"Error resampling chunk {i}: {resample_e}. Appending zeros.")
        upsampled_chunks.append(np.zeros(num_samples_chunk_recon, dtype=complex))

# --- 5. Wavelet-Based Processing AT HIGH RATE ---
print("\n--- Performing Wavelet Processing AT RECONSTRUCTION RATE ---")

total_duration_recon = sum(m['intended_duration_s'] for m in loaded_metadata) + \
                      global_attrs['tuning_delay_s'] * (len(loaded_chunks) - 1)
num_samples_recon = int(round(total_duration_recon * fs_recon))
reconstructed_signal = np.zeros(num_samples_recon, dtype=complex)
sum_of_windows = np.zeros(num_samples_recon, dtype=float)

print(f"Reconstruction target: {num_samples_recon} samples @ {fs_recon/1e6:.2f} MHz")

# Process each UPSAMPLED chunk with wavelets
current_recon_time = 0.0
for i, up_chunk in tqdm(enumerate(upsampled_chunks), total=len(upsampled_chunks), desc="Wavelet Recon"):
    meta = loaded_metadata[i]
    cp_phase_estimate = estimated_chunk_phases[i]

    if len(up_chunk) < pywt.Wavelet(wavelet_name).dec_len * (2**wpd_level): # Check minimum length for WPD
        print(f"Warning: Upsampled chunk {i} too short ({len(up_chunk)}) for WPD level {wpd_level}. Skipping.")
        current_recon_time += meta['intended_duration_s'] + global_attrs['tuning_delay_s']
        continue

    try:
        # Perform WPD on the upsampled chunk
        wp = pywt.WaveletPacket(data=up_chunk, wavelet=wavelet_name, mode='symmetric', maxlevel=wpd_level)

        # Apply phase correction to leaf nodes at the desired level
        nodes = wp.get_level(wpd_level, order='natural', decompose=False) # Get nodes without forcing decomp
        for node in nodes:
            # Ensure we are at the correct level (sometimes get_level returns shallower nodes)
            if node.level == wpd_level:
                 # Apply phase correction
                 node.data = node.data * np.exp(-1j * cp_phase_estimate)


        # Reconstruct the phase-corrected chunk AT THE HIGH RATE
        # update=False uses the modified coefficients
        reconstructed_chunk = wp.reconstruct(update=False)

        # Length should match the upsampled chunk length
        if len(reconstructed_chunk) != len(up_chunk):
             print(f"Warning: Wavelet reconstructed chunk {i} length ({len(reconstructed_chunk)}) "
                   f"differs from upsampled length ({len(up_chunk)}). Adjusting.")
             # Trim or pad as needed to match expected upsampled length for stitching
             expected_len = len(up_chunk)
             if len(reconstructed_chunk) > expected_len:
                 reconstructed_chunk = reconstructed_chunk[:expected_len]
             else:
                 reconstructed_chunk = np.pad(reconstructed_chunk, (0, expected_len - len(reconstructed_chunk)))

    except Exception as wp_e:
        print(f"Error during Wavelet processing for chunk {i}: {wp_e}. Skipping chunk.")
        current_recon_time += meta['intended_duration_s'] + global_attrs['tuning_delay_s']
        continue


    # --- Stitching (Overlap-Add) ---
    start_idx_recon = int(round(current_recon_time * fs_recon))
    end_idx_recon = min(start_idx_recon + len(reconstructed_chunk), num_samples_recon)
    actual_len = end_idx_recon - start_idx_recon
    if actual_len <= 0:
        current_recon_time += meta['intended_duration_s'] + global_attrs['tuning_delay_s']
        continue

    window = sig.get_window(stitching_window_type, actual_len)
    # Ensure window length matches signal slice length
    if len(window) != len(reconstructed_chunk[:actual_len]):
        print(f"Warning: Window length mismatch in chunk {i}. Adjusting window.")
        window = sig.get_window(stitching_window_type, len(reconstructed_chunk[:actual_len]))

    reconstructed_signal[start_idx_recon:end_idx_recon] += reconstructed_chunk[:actual_len] * window
    sum_of_windows[start_idx_recon:end_idx_recon] += window

    # Update time offset for the next chunk
    current_recon_time += meta['intended_duration_s'] + global_attrs['tuning_delay_s']


# Avoid division by zero where window sum is zero
sum_of_windows[sum_of_windows < 1e-9] = 1.0
# Normalize the reconstructed signal by the sum of windows
reconstructed_signal /= sum_of_windows

print("\nWavelet reconstruction and stitching complete.")

# --- 6. Evaluation & Visualization ---
# (Identical to the previous script - assuming no changes needed here)
print("\n--- Evaluating Reconstruction ---")

print("Regenerating ground truth baseband for comparison...")
gt_duration = total_duration_recon
num_samples_gt_compare = int(round(gt_duration * fs_recon))
t_gt_compare = np.linspace(0, gt_duration, num_samples_gt_compare, endpoint=False)
gt_baseband_compare = np.zeros(num_samples_gt_compare, dtype=complex)

mod = global_attrs['modulation']
bw_gt = global_attrs['total_signal_bandwidth_hz']

if mod.lower() == 'qam16':
    symbol_rate_gt = bw_gt / 4
    num_symbols_gt = int(np.ceil(gt_duration * symbol_rate_gt))
    symbols_real = np.random.choice([-3, -1, 1, 3], size=num_symbols_gt)
    symbols_imag = np.random.choice([-3, -1, 1, 3], size=num_symbols_gt)
    symbols = (symbols_real + 1j * symbols_imag) / np.sqrt(10)
    samples_per_symbol_gt = int(fs_recon / symbol_rate_gt)
    if samples_per_symbol_gt == 0:
        samples_per_symbol_gt = 1
    baseband_symbols = np.repeat(symbols, samples_per_symbol_gt)
    len_to_take = min(len(baseband_symbols), num_samples_gt_compare)
    gt_baseband_compare[:len_to_take] = baseband_symbols[:len_to_take]
else:
    print(f"Warning: Ground truth regeneration not implemented for modulation '{mod}'. Using zeros.")

# Normalize comparison ground truth
mean_power_gt = np.mean(np.abs(gt_baseband_compare)**2)
if mean_power_gt > 1e-20:
    gt_baseband_compare /= np.sqrt(mean_power_gt)

# Align reconstructed signal amplitude
mean_power_recon = np.mean(np.abs(reconstructed_signal)**2)
if mean_power_recon > 1e-20:
    reconstructed_signal_aligned = reconstructed_signal * (np.sqrt(mean_power_gt) / np.sqrt(mean_power_recon))
else:
    reconstructed_signal_aligned = reconstructed_signal

# Calculate Mean Squared Error (MSE)
# Check for NaNs/Infs before MSE calculation
if not np.all(np.isfinite(reconstructed_signal_aligned)):
     print("Error: Reconstructed signal contains non-finite values. Cannot calculate MSE.")
     mse = np.inf
else:
     mse = np.mean(np.abs(gt_baseband_compare - reconstructed_signal_aligned)**2)
print(f"Mean Squared Error (MSE) between Ground Truth and Reconstructed: {mse:.4e}")

# Plotting
plt.style.use('seaborn-v0_8-darkgrid')
fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Plot Ground Truth (Baseband)
time_axis_plot = t_gt_compare[:plot_length] * 1e6
axs[0].plot(time_axis_plot, gt_baseband_compare[:plot_length].real, label='Ground Truth (Real)')
axs[0].plot(time_axis_plot, gt_baseband_compare[:plot_length].imag, label='Ground Truth (Imag)', alpha=0.7)
axs[0].set_title('Ground Truth Baseband Signal')
axs[0].set_ylabel('Amplitude')
axs[0].legend(fontsize='small')
axs[0].grid(True)

# Plot Reconstructed Signal
# Add check for non-finite values before plotting
plot_data_recon = reconstructed_signal_aligned[:plot_length]
plot_data_recon[~np.isfinite(plot_data_recon)] = 0 # Replace non-finite with 0 for plot
axs[1].plot(time_axis_plot, plot_data_recon.real, label='Reconstructed (Real)')
axs[1].plot(time_axis_plot, plot_data_recon.imag, label='Reconstructed (Imag)', alpha=0.7)
axs[1].set_title(f'Reconstructed Signal (MSE: {mse:.2e})')
axs[1].set_ylabel('Amplitude')
axs[1].legend(fontsize='small')
axs[1].grid(True)

# Plot Error Signal
error_signal = gt_baseband_compare[:plot_length] - plot_data_recon # Use the potentially modified plot_data
axs[2].plot(time_axis_plot, error_signal.real, label='Error (Real)')
axs[2].plot(time_axis_plot, error_signal.imag, label='Error (Imag)', alpha=0.7)
axs[2].set_title('Reconstruction Error')
axs[2].set_xlabel('Time (µs)')
axs[2].set_ylabel('Amplitude')
axs[2].legend(fontsize='small')
axs[2].grid(True)

plt.tight_layout()
plt.show()

# Plot Spectra Comparison
plt.figure(figsize=(12, 7))
# Add checks for sufficient length and finite values before FFT
if len(reconstructed_signal_aligned) > 1 and np.all(np.isfinite(reconstructed_signal_aligned)):
    f_recon_axis = np.fft.fftshift(np.fft.fftfreq(len(reconstructed_signal_aligned), d=1/fs_recon))
    spec_recon = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(reconstructed_signal_aligned))) + 1e-12)
    spec_recon -= np.nanmax(spec_recon) # Use nanmax in case NaNs crept in somehow
    plt.plot(f_recon_axis / 1e6, spec_recon, label='Reconstructed Spectrum', linestyle='--', alpha=0.8)
else:
    print("Skipping reconstructed spectrum plot due to insufficient data or non-finite values.")


if len(gt_baseband_compare) > 1:
    f_gt_axis = np.fft.fftshift(np.fft.fftfreq(len(gt_baseband_compare), d=1/fs_recon))
    spec_gt = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(gt_baseband_compare))) + 1e-12)
    spec_gt -= np.nanmax(spec_gt) # Use nanmax
    plt.plot(f_gt_axis / 1e6, spec_gt, label='Ground Truth Spectrum', alpha=0.8)
else:
     print("Skipping ground truth spectrum plot due to insufficient data.")


plt.title('Spectra Comparison (Baseband)')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude (dB rel. Max)')
plt.ylim(bottom=spectrum_ylim)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nScript finished.")