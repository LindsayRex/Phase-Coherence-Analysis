import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import h5py
from tqdm import tqdm
import sys
import pywt  # For wavelet transforms

# --- Parameters ---
# Use the RMS_FIXED data file!
input_filename = "simulated_chunks_25GHz_400MHzBW_qam16_sdr56MHz.h5"

stitching_window_type = 'hann'
plot_length = 5000
spectrum_ylim = -100
EXPECTED_RMS = 1.39e-02 # Target RMS after initial scaling

# --- WPD Phase Correction Parameters ---
wpd_wavelet = 'db4'  # Wavelet for WPD
wpd_level = 4      # Decomposition level for WPD (adjust based on needs/computation)
# Note: Higher levels give finer frequency but shorter coefficient sequences

# --- 1. Load Simulated Chunk Data ---
print(f"Loading data from: {input_filename}")
loaded_chunks = []
loaded_metadata = []
global_attrs = {}
try:
    with h5py.File(input_filename, 'r') as f:
        # --- Metadata Verification ---
        print("\n--- Verifying Loaded Metadata Phase ---")
        true_phases_rad = []
        actual_chunks_meta = f.attrs.get('actual_num_chunks_saved', 0)
        for i in range(actual_chunks_meta):
             group_name = f'chunk_{i:03d}'
             if group_name in f:
                 meta_group = f[group_name]
                 phase_rad = meta_group.attrs.get('applied_phase_offset_rad', np.nan)
                 true_phases_rad.append(phase_rad)
                 phase_deg = np.rad2deg(phase_rad) if np.isfinite(phase_rad) else 'MISSING or NaN'
                 print(f"Chunk {i}: loaded applied_phase_offset_rad = {phase_rad} (approx {phase_deg} deg)")
             else:
                  print(f"Chunk {i}: Group not found during metadata check.")
                  true_phases_rad.append(np.nan) # Append NaN if chunk missing

        # --- Load Global Attrs and Chunks ---
        for key, value in f.attrs.items(): global_attrs[key] = value
        print("\n--- Global Parameters ---")
        for key, value in global_attrs.items(): print(f"{key}: {value}")
        print("-------------------------")
        actual_chunks = global_attrs.get('actual_num_chunks_saved', 0)
        if actual_chunks == 0: raise ValueError("No chunks.")
        for i in range(actual_chunks):
            group = f[f'chunk_{i:03d}']
            chunk_data = group['iq_data'][:].astype(np.complex128)
            meta = {key: value for key, value in group.attrs.items()} # Load full meta again
            loaded_chunks.append(chunk_data)
            loaded_metadata.append(meta) # Store full metadata dictionary
except Exception as e:
    print(f"Error loading HDF5 file '{input_filename}': {e}"); sys.exit(1)
if not loaded_chunks: print("No chunk data loaded. Exiting."); sys.exit(1)
print(f"\nSuccessfully loaded {len(loaded_chunks)} chunks.")

# Define global constants from loaded attributes
GLBL_SDR_SAMPLE_RATE = global_attrs.get('sdr_sample_rate_hz', None)
GLBL_RECON_SAMPLE_RATE = global_attrs.get('ground_truth_sample_rate_hz', None)
GLBL_OVERLAP_FACTOR = global_attrs.get('overlap_factor', 0.1)
GLBL_TUNING_DELAY = global_attrs.get('tuning_delay_s', 5e-6)
# SYMBOL_RATE = global_attrs['total_signal_bandwidth_hz'] / 4 # Not needed directly here

# Validate the loaded parameters
if GLBL_SDR_SAMPLE_RATE is None or GLBL_RECON_SAMPLE_RATE is None: print("Error: Sample rate information missing."); sys.exit(1)
if GLBL_SDR_SAMPLE_RATE <= 0 or GLBL_RECON_SAMPLE_RATE <= 0: print("Error: Invalid sample rates."); sys.exit(1)

# --- 1b. Correct Initial Amplitude Scaling ---
print("\n--- Correcting Initial Amplitude Scaling ---")
scaled_loaded_chunks = []
print("Chunk | RMS Before | Scaling Factor | RMS After  | Max Abs (After)")
print("------|------------|----------------|------------|----------------")
scaling_successful = True
for i, chunk in enumerate(loaded_chunks):
    # ... (Scaling code remains exactly the same as before) ...
    if len(chunk) == 0:
        scaled_loaded_chunks.append(chunk); print(f"{i:<5d} | --- EMPTY ---    | ---            | --- EMPTY ---    | ---"); continue
    if not np.all(np.isfinite(chunk)):
        print(f"ERROR: Chunk {i} non-finite BEFORE scaling."); scaling_successful = False; scaled_loaded_chunks.append(chunk); continue
    rms_before_scaling = np.sqrt(np.mean(np.abs(chunk)**2))
    max_abs_before = np.max(np.abs(chunk)) if len(chunk)>0 else 0
    if rms_before_scaling < 1e-12:
        print(f"{i:<5d} | {rms_before_scaling:.4e}       | SKIPPED (Zero) | {rms_before_scaling:.4e}       | {max_abs_before:.4e}"); scaled_loaded_chunks.append(chunk); continue
    scaling_factor = EXPECTED_RMS / rms_before_scaling
    scaled_chunk = (chunk * scaling_factor).astype(np.complex128)
    if not np.all(np.isfinite(scaled_chunk)):
         print(f"ERROR: Chunk {i} non-finite AFTER scaling."); scaling_successful = False; scaled_loaded_chunks.append(scaled_chunk); continue
    rms_after_scaling = np.sqrt(np.mean(np.abs(scaled_chunk)**2))
    max_abs_after = np.max(np.abs(scaled_chunk))
    if not np.isclose(rms_after_scaling, EXPECTED_RMS, rtol=1e-3): print(f"WARNING: Chunk {i} RMS scaling mismatch ({rms_after_scaling:.4e} vs {EXPECTED_RMS:.4e})")
    print(f"{i:<5d} | {rms_before_scaling:.4e} | {scaling_factor:<14.4f} | {rms_after_scaling:.4e} | {max_abs_after:.4e}")
    scaled_loaded_chunks.append(scaled_chunk)
if not scaling_successful: print("\nERROR: Non-finite values detected during scaling. Cannot proceed."); sys.exit(1)
print("--- Initial Amplitude Scaling Complete ---")

# --- Adaptive Filtering Pre-Alignment (BYPASSED) ---
print("\n--- SKIPPING Adaptive Filtering Pre-Alignment (FOR DEBUGGING) ---")
aligned_chunks = scaled_loaded_chunks # Use scaled chunks directly


# --- 2. Upsample Chunks (FFT Method) ---
# Using the manual FFT zero-padding method from bigdaddy3.py
print("\n--- Upsampling chunks (using FFT method) ---")
upsampled_chunks = []
# ... (Upsampling code block remains exactly the same as in bigdaddy3.py) ...
debug_rms_upsample_input = [] # Keep debug lists if needed
debug_rms_upsample_output = []
debug_max_abs_upsample_input = []
debug_max_abs_upsample_output = []
for i, chunk_data in tqdm(enumerate(aligned_chunks), total=len(aligned_chunks), desc="Upsampling"):
    chunk_duration = len(chunk_data)/GLBL_SDR_SAMPLE_RATE if GLBL_SDR_SAMPLE_RATE > 0 else 0
    num_samples_chunk_recon = int(round(chunk_duration * GLBL_RECON_SAMPLE_RATE))
    if len(chunk_data) < 2:
        upsampled_chunks.append(np.zeros(num_samples_chunk_recon, dtype=complex)); continue # Simplified append for brevity
    try:
        rms_in = np.sqrt(np.mean(np.abs(chunk_data)**2))
        max_abs_in = np.max(np.abs(chunk_data))
        debug_rms_upsample_input.append(rms_in)
        debug_max_abs_upsample_input.append(max_abs_in)
        if rms_in < 1e-12:
            upsampled_chunks.append(np.zeros(num_samples_chunk_recon, dtype=complex)); continue
        n_orig = len(chunk_data); n_new = num_samples_chunk_recon
        fft_orig = np.fft.fft(chunk_data); fft_shifted = np.fft.fftshift(fft_orig)
        fft_padded = np.zeros(n_new, dtype=complex); start_idx = (n_new - n_orig) // 2
        fft_padded[start_idx:start_idx+n_orig] = fft_shifted
        fft_padded = np.fft.ifftshift(fft_padded)
        upsampled_chunk = np.fft.ifft(fft_padded) * (n_new / n_orig)
        upsampled_chunk = upsampled_chunk.astype(np.complex128)
        rms_out = np.sqrt(np.mean(np.abs(upsampled_chunk)**2))
        max_abs_out = np.max(np.abs(upsampled_chunk))
        debug_rms_upsample_output.append(rms_out)
        debug_max_abs_upsample_output.append(max_abs_out)
        upsampled_chunks.append(upsampled_chunk.copy())
        # --- Plotting first chunk (Optional but keep for now) ---
        if i == 0:
             print("Plotting first upsampled chunk (FFT method)...")
             plt.figure(figsize=(12,4)); time_axis_debug = np.arange(len(upsampled_chunk))/GLBL_RECON_SAMPLE_RATE*1e6
             plt.plot(time_axis_debug, upsampled_chunk.real, label='Real'); plt.plot(time_axis_debug, upsampled_chunk.imag, label='Imag', alpha=0.7)
             plt.title(f'First Upsampled Chunk (RMS={rms_out:.3e})'); plt.xlabel('Time (µs)'); plt.ylabel('Amp'); plt.legend(); plt.grid(True);
             ylim_abs = max(max_abs_out * 1.2 if np.isfinite(max_abs_out) else 0.1, 0.05)
             plt.ylim(-ylim_abs, ylim_abs); plt.xlim(-0.1, 5.0); plt.show()
             # print("Plotting Spectra Comparison (After Upsampling)...") # Can skip spectrum plot here
             # ... spectrum plotting code ...
    except Exception as resample_e:
        print(f"Error resampling chunk {i}: {resample_e}. Appending zeros.")
        upsampled_chunks.append(np.zeros(num_samples_chunk_recon, dtype=complex))
# ... (Print RMS table if desired) ...


# --- 3. Phase Correction using Time-Domain Correlation ---
print("\n--- Performing Time-Domain Phase Correction ---")

# Calculate overlap samples at reconstruction rate
chunk_duration_s = len(loaded_chunks[0]) / GLBL_SDR_SAMPLE_RATE if GLBL_SDR_SAMPLE_RATE > 0 else 0
overlap_samples_recon = int(round(chunk_duration_s * GLBL_OVERLAP_FACTOR * GLBL_RECON_SAMPLE_RATE))
print(f"Overlap samples at Recon Rate: {overlap_samples_recon}")

# Initialize cumulative phases
estimated_cumulative_phases = [0.0]  # Chunk 0 is reference
phase_corrected_chunks = [upsampled_chunks[0]]  # First chunk unchanged

print("Estimating phase differences and applying corrections...")
for i in tqdm(range(1, len(upsampled_chunks)), desc="Phase Correction"):
    if len(upsampled_chunks[i-1]) < overlap_samples_recon or len(upsampled_chunks[i]) < overlap_samples_recon:
        print(f"Warning: Insufficient overlap for chunk {i}. Skipping correction.")
        estimated_cumulative_phases.append(estimated_cumulative_phases[-1])
        phase_corrected_chunks.append(upsampled_chunks[i])
        continue

    # Extract overlapping segments
    overlap_prev = upsampled_chunks[i-1][-overlap_samples_recon:]
    overlap_curr = upsampled_chunks[i][:overlap_samples_recon]

    # Compute time-domain correlation
    sum_corr = np.sum(overlap_curr * np.conj(overlap_prev))

    # Estimate phase difference
    if np.abs(sum_corr) > 1e-12:  # Avoid division by near-zero
        delta_phi = np.angle(sum_corr)  # phi_i - phi_{i-1}
    else:
        print(f"Warning: Correlation sum too small for chunk {i}. Using zero phase difference.")
        delta_phi = 0.0

    # Update cumulative phase
    prev_cumulative = estimated_cumulative_phases[-1]
    current_cumulative_phase = prev_cumulative + delta_phi
    estimated_cumulative_phases.append(current_cumulative_phase)

    # Apply phase correction
    corrected_chunk = upsampled_chunks[i] * np.exp(-1j * current_cumulative_phase)
    phase_corrected_chunks.append(corrected_chunk)

print("--- Time-Domain Phase Correction Complete ---")

# --- 4. Stitching (Overlap-Add) ---
# This section now uses 'phase_corrected_chunks' from the WPD step
print("\n--- Performing Stitching (using WPD Phase Corrected Chunks) ---")
# Calculate necessary parameters for stitching buffer
# Use the same calculation as bigdaddy3.py
time_advance_per_chunk = chunk_duration_s * (1.0 - GLBL_OVERLAP_FACTOR)
total_duration_recon = chunk_duration_s + (len(loaded_chunks) - 1) * (time_advance_per_chunk + GLBL_TUNING_DELAY)
num_samples_recon = int(round(total_duration_recon * GLBL_RECON_SAMPLE_RATE))

if num_samples_recon <= 0: print(f"Error: Invalid num_samples_recon {num_samples_recon}"); sys.exit(1)

reconstructed_signal = np.zeros(num_samples_recon, dtype=complex)
sum_of_windows = np.zeros(num_samples_recon, dtype=float)

print(f"Reconstruction target buffer: {num_samples_recon} samples @ {GLBL_RECON_SAMPLE_RATE/1e6:.2f} MHz (Est. Duration: {total_duration_recon*1e6:.1f} us)")

current_recon_time_start = 0.0
# plt.figure(figsize=(14, 8)) # Optional progressive plot
# num_subplot_rows = int(np.ceil(len(phase_corrected_chunks)/2))

for i, chunk_to_add in tqdm(enumerate(phase_corrected_chunks), total=len(phase_corrected_chunks), desc="Stitching"):
    # --- Stitching logic remains the same, just input is different ---
    if i >= len(loaded_metadata): # Ensure metadata index valid for time advance calc
        if i < len(phase_corrected_chunks) - 1: current_recon_time_start += time_advance_per_chunk + GLBL_TUNING_DELAY
        continue
    meta = loaded_metadata[i] # Needed only if using meta['intended_duration_s']? Sticking to fixed advance.

    if len(chunk_to_add) == 0:
        if i < len(phase_corrected_chunks) - 1: current_recon_time_start += time_advance_per_chunk + GLBL_TUNING_DELAY
        continue

    start_idx_recon = int(round(current_recon_time_start * GLBL_RECON_SAMPLE_RATE))
    num_samples_in_chunk = len(chunk_to_add)
    end_idx_recon = min(start_idx_recon + num_samples_in_chunk, num_samples_recon)
    actual_len = end_idx_recon - start_idx_recon
    if actual_len <= 0:
        if i < len(phase_corrected_chunks) - 1: current_recon_time_start += time_advance_per_chunk + GLBL_TUNING_DELAY
        continue

    if actual_len < 2: window = np.ones(actual_len)
    else: window = sig.get_window(stitching_window_type, actual_len)
        # Normalization - use consistent method (e.g., RMS=1 or sum=1 based on window)
        # window /= np.sqrt(np.mean(window**2)) # RMS=1 approx
        # Or normalize such that sum of squares in overlap is approx 1 (depends on window & overlap)
        # Let's use simple RMS=1 normalization for now
    window_rms = np.sqrt(np.mean(window**2))
    if window_rms > 1e-9: window /= window_rms

    try:
        segment_to_add = chunk_to_add[:actual_len] * window
        if not np.all(np.isfinite(segment_to_add)): segment_to_add[~np.isfinite(segment_to_add)] = 0
        reconstructed_signal[start_idx_recon:end_idx_recon] += segment_to_add
        sum_of_windows[start_idx_recon:end_idx_recon] += window # Sum unnormalized window for divisor? Or normalized? Let's sum normalized.
        if not np.all(np.isfinite(reconstructed_signal[start_idx_recon:end_idx_recon])): reconstructed_signal[start_idx_recon:end_idx_recon] = 0
    except Exception as add_e: print(f"Error overlap-add chunk {i}: {add_e}"); continue

    # Update time for next chunk
    if i < len(phase_corrected_chunks) - 1:
        current_recon_time_start += time_advance_per_chunk + GLBL_TUNING_DELAY

# Optional progressive plot showing
# if i == 0 or (i + 1) % max(1, len(phase_corrected_chunks)//4) == 0 : # Plot occasionally
#      plt.clf(); plt.plot(np.abs(reconstructed_signal)); plt.pause(0.1)
# plt.close() # Close progressive plot

print("\nStitching loop complete.")

# --- Normalization Step ---
# (Normalization code remains the same)
print("\n--- Normalizing reconstructed signal ---")
rms_before_norm = np.sqrt(np.mean(np.abs(reconstructed_signal)**2)) if np.all(np.isfinite(reconstructed_signal)) else np.nan
max_abs_before_norm = np.nanmax(np.abs(reconstructed_signal)) if len(reconstructed_signal) > 0 and np.all(np.isfinite(reconstructed_signal)) else np.nan
print("Signal Stats BEFORE Normalization:")
print(f"  RMS: {rms_before_norm:.4e}")
print(f"  Max Abs: {max_abs_before_norm:.4e}")
reliable_threshold = 1e-6
reliable_indices = np.where(sum_of_windows >= reliable_threshold)[0]
fallback_sum = 1.0
if len(reliable_indices) > 0:
    median_sum = np.median(sum_of_windows[reliable_indices])
    if np.isfinite(median_sum) and median_sum > 1e-9: fallback_sum = median_sum
print(f"Using median sum_of_windows ({fallback_sum:.4f}) as fallback divisor.")
sum_of_windows_divisor = sum_of_windows.copy()
sum_of_windows_divisor[sum_of_windows < reliable_threshold] = fallback_sum
sum_of_windows_divisor[np.abs(sum_of_windows_divisor) < 1e-15] = fallback_sum # Ensure no zeros
reconstructed_signal_normalized = np.zeros_like(reconstructed_signal)
valid_divisor = np.abs(sum_of_windows_divisor) > 1e-15
np.divide(reconstructed_signal, sum_of_windows_divisor, out=reconstructed_signal_normalized, where=valid_divisor)
if not np.all(np.isfinite(reconstructed_signal_normalized)):
     print("Warning: Non-finite values after normalization. Zeroing.")
     reconstructed_signal_normalized[~np.isfinite(reconstructed_signal_normalized)] = 0
rms_after_norm = np.sqrt(np.mean(np.abs(reconstructed_signal_normalized)**2))
max_abs_after_norm = np.nanmax(np.abs(reconstructed_signal_normalized)) if len(reconstructed_signal_normalized) > 0 else 0
print("\nSignal Stats AFTER Normalization:")
print(f"  RMS: {rms_after_norm:.4e}")
print(f"  Max Abs: {max_abs_after_norm:.4e}")
print(f"  Ratio to Initial Target RMS ({EXPECTED_RMS:.4e}): {rms_after_norm/EXPECTED_RMS:.4f}")
reconstructed_signal = reconstructed_signal_normalized.copy()
print("\nNormalization complete.")

# --- 5. Evaluation & Visualization ---
# (Evaluation and plotting code remains the same)
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
else: print(f"Warning: GT regen not implemented for {mod}")

gt_rms_before = np.sqrt(np.mean(np.abs(gt_baseband_compare)**2)); print(f"GT RMS Before Scale: {gt_rms_before:.4e}")
if gt_rms_before > 1e-20:
    gt_scale_factor = EXPECTED_RMS / gt_rms_before; gt_baseband_compare *= gt_scale_factor
    gt_rms_after = np.sqrt(np.mean(np.abs(gt_baseband_compare)**2)); print(f"Scaled GT baseband. Target RMS: {EXPECTED_RMS:.4e}, Actual RMS: {gt_rms_after:.4e}")
else: print("Warning: GT baseband power too low. Skipping scaling.")

def compute_spectrum(signal, sample_rate):
    """
    Compute the power spectrum of a signal in dB.
    
    Args:
        signal (np.ndarray): Input signal
        sample_rate (float): Sampling frequency in Hz
    
    Returns:
        freqs (np.ndarray): Frequency array
        spec_db (np.ndarray): Power spectrum in dB
    """
    n = len(signal)
    freqs = np.fft.fftfreq(n, d=1/sample_rate)
    fft_result = np.fft.fft(signal)
    spec = np.abs(fft_result)**2 / n  # Power spectrum
    spec_db = 10 * np.log10(spec + 1e-12)  # Convert to dB, avoid log(0)
    return freqs, spec_db


# Evaluation
mse = np.inf; nmse = np.inf; evm = np.inf
reconstructed_signal_aligned = reconstructed_signal # Use non-aligned for plot alignment calc
if len(reliable_indices) >= 2 and np.all(np.isfinite(reconstructed_signal[reliable_indices])) and np.all(np.isfinite(gt_baseband_compare[reliable_indices])):
    print(f"Evaluating metrics using {len(reliable_indices)} reliable samples.")
    gt_reliable = gt_baseband_compare[reliable_indices]
    recon_reliable = reconstructed_signal[reliable_indices]
    mean_power_gt_reliable = np.mean(np.abs(gt_reliable)**2)
    mean_power_recon_reliable = np.mean(np.abs(recon_reliable)**2)
    if mean_power_gt_reliable > 1e-20 and mean_power_recon_reliable > 1e-20:
        error_reliable = gt_reliable - recon_reliable * np.sqrt(mean_power_gt_reliable / mean_power_recon_reliable) # Error relative to scaled recon
        mse = np.mean(np.abs(error_reliable)**2)
        nmse = mse / mean_power_gt_reliable
        evm = np.sqrt(nmse) * 100 if nmse >=0 else np.inf
        # Align for plotting
        power_scale_factor = np.sqrt(mean_power_gt_reliable / mean_power_recon_reliable)
        reconstructed_signal_aligned = reconstructed_signal * power_scale_factor # Align the whole signal
        print(f"Applied plotting alignment scale factor: {power_scale_factor:.4f}")
    else: print("Warning: Zero power in reliable segments, cannot calculate metrics/align.")
else: print(f"Warning: Not enough reliable samples ({len(reliable_indices)}) or non-finite values. Skipping metric calculation.")


print("\nEvaluation Metrics:")
print(f"  MSE : {mse:.4e}")
print(f"  NMSE: {nmse:.4e} ({10*np.log10(nmse):.2f} dB)" if np.isfinite(nmse) else "  NMSE: Inf / NaN")
print(f"  EVM : {evm:.2f}%" if np.isfinite(evm) else "  EVM : Inf / NaN")

# --- Plotting ---
# (Plotting code remains the same)
print("\n--- Generating Matplotlib Plots ---")
plt.style.use('seaborn-v0_8-darkgrid')
fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
# Use GT scaled to EXPECTED_RMS for plotting consistency
gt_plot_data = gt_baseband_compare
plot_samples = min(plot_length, len(t_gt_compare), len(gt_plot_data))
if plot_samples > 0:
    time_axis_plot = t_gt_compare[:plot_samples] * 1e6
    axs[0].plot(time_axis_plot, gt_plot_data[:plot_samples].real, label='GT (Real)')
    axs[0].plot(time_axis_plot, gt_plot_data[:plot_samples].imag, label='GT (Imag)', alpha=0.7)
    axs[0].set_title(f'Ground Truth (Target RMS={EXPECTED_RMS:.2e})'); axs[0].set_ylabel('Amp'); axs[0].legend(fontsize='small'); axs[0].grid(True)
    ylim_gt = max(np.max(np.abs(gt_plot_data[:plot_samples]))*1.2, 0.01) if len(gt_plot_data)>0 else 0.01; axs[0].set_ylim(-ylim_gt, ylim_gt)

    plot_samples_recon = min(plot_length, len(reconstructed_signal_aligned))
    plot_data_recon = reconstructed_signal_aligned[:plot_samples_recon]
    plot_data_recon[~np.isfinite(plot_data_recon)] = 0
    axs[1].plot(time_axis_plot[:plot_samples_recon], plot_data_recon.real, label='Recon (Real)')
    axs[1].plot(time_axis_plot[:plot_samples_recon], plot_data_recon.imag, label='Recon (Imag)', alpha=0.7)
    axs[1].set_title(f'Reconstructed (Aligned, EVM: {evm:.2f}%)'); axs[1].set_ylabel('Amp'); axs[1].legend(fontsize='small'); axs[1].grid(True)
    axs[1].set_ylim(axs[0].get_ylim()) # Match GT ylim

    plot_samples_error = min(plot_samples, plot_samples_recon)
    error_signal = gt_plot_data[:plot_samples_error] - plot_data_recon[:plot_samples_error] # Error between GT and ALIGNED Recon
    error_signal[~np.isfinite(error_signal)] = 0
    axs[2].plot(time_axis_plot[:plot_samples_error], error_signal.real, label='Error (Real)')
    axs[2].plot(time_axis_plot[:plot_samples_error], error_signal.imag, label='Error (Imag)', alpha=0.7)
    axs[2].set_title('Error (GT - Recon Aligned)'); axs[2].set_xlabel('Time (µs)'); axs[2].set_ylabel('Amplitude'); axs[2].legend(fontsize='small'); axs[2].grid(True)
    ylim_err = max(np.max(np.abs(error_signal))*1.2, 0.01) if len(error_signal)>0 else 0.01; axs[2].set_ylim(-ylim_err, ylim_err)

    plt.tight_layout(); plt.show()

# --- Spectrum Plot ---
plt.figure(figsize=(12, 7))
plot_spectrum_flag = False
if len(reconstructed_signal_aligned) > 1 and np.all(np.isfinite(reconstructed_signal_aligned)):
    f_recon, spec_recon_db = compute_spectrum(reconstructed_signal_aligned, GLBL_RECON_SAMPLE_RATE)
    if len(f_recon)>0: plt.plot(f_recon/1e6, spec_recon_db, label='Recon Spec (Aligned)', ls='--', alpha=0.8); plot_spectrum_flag = True
if len(gt_baseband_compare) > 1 and np.all(np.isfinite(gt_baseband_compare)):
    f_gt, spec_gt_db = compute_spectrum(gt_baseband_compare, GLBL_RECON_SAMPLE_RATE)
    if len(f_gt)>0: plt.plot(f_gt/1e6, spec_gt_db, label='GT Spec', alpha=0.8); plot_spectrum_flag = True
if plot_spectrum_flag:
    plt.title('Spectra Comparison'); plt.xlabel('Freq (MHz)'); plt.ylabel('Mag (dB)'); plt.ylim(spectrum_ylim, 5); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
else: print("Skipping spectrum plot.")

print("\nScript finished.")

