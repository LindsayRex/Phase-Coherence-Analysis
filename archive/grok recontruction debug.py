import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import h5py
from tqdm import tqdm
import sys # Needed for exit()

# --- Parameters ---
input_filename = "simulated_chunks_25GHz_400MHzBW_qam16_sdr56MHz.h5"
# stitching_window_type = 'hann' # Not used in this version
plot_length = 5000
spectrum_ylim = -100
EXPECTED_RMS = 1.0 # Target RMS after initial scaling

# --- 1. Load Simulated Chunk Data ---
# (Loading code remains the same)
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
print(f"\nLoaded {len(loaded_chunks)} chunks.")
fs_sdr = global_attrs.get('sdr_sample_rate_hz', None)
fs_recon = global_attrs.get('ground_truth_sample_rate_hz', None)
overlap_factor = global_attrs.get('overlap_factor', 0.1)
tuning_delay = global_attrs.get('tuning_delay_s', 5e-6)
if fs_sdr is None or fs_recon is None:
     print("Error: Sample rate information missing."); sys.exit(1)

# --- 1b. Correct Initial Amplitude Scaling ---
# (Scaling code remains the same)
print("\n--- Correcting Initial Amplitude Scaling of Loaded Chunks ---")
scaled_loaded_chunks = []
print("Chunk | RMS Before Scaling | Scaling Factor | RMS After Scaling")
print("------|------------------|----------------|------------------")
scaling_successful = True
for i, chunk in enumerate(loaded_chunks):
    if len(chunk) == 0: scaled_loaded_chunks.append(chunk); print(f"{i:<5d} | --- EMPTY ---    | ---            | --- EMPTY ---"); continue
    if not np.all(np.isfinite(chunk)): print(f"ERROR: Chunk {i} non-finite BEFORE scaling."); scaling_successful = False; scaled_loaded_chunks.append(chunk); continue
    rms_before_scaling = np.sqrt(np.mean(np.abs(chunk)**2))
    if rms_before_scaling < 1e-12: print(f"{i:<5d} | {rms_before_scaling:.4e}       | SKIPPED (Zero) | {rms_before_scaling:.4e}"); scaled_loaded_chunks.append(chunk); continue
    scaling_factor = EXPECTED_RMS / rms_before_scaling
    scaled_chunk = (chunk * scaling_factor).astype(np.complex128)
    rms_after_scaling = np.sqrt(np.mean(np.abs(scaled_chunk)**2))
    if not np.isclose(rms_after_scaling, EXPECTED_RMS, rtol=1e-3): print(f"WARNING: Chunk {i} RMS scaling mismatch ({rms_after_scaling:.4e} vs {EXPECTED_RMS:.4e})")
    else: print(f"{i:<5d} | {rms_before_scaling:.4e}       | {scaling_factor:<14.4f} | {rms_after_scaling:.4e}")
    scaled_loaded_chunks.append(scaled_chunk)
if not scaling_successful: print("\nERROR: Non-finite values detected. Cannot proceed."); sys.exit(1)
print("--- Initial Amplitude Scaling Complete ---")

# --- SKIPPING AF ---
print("\n--- Skipping Adaptive Filtering ---")
aligned_chunks = scaled_loaded_chunks

# --- SKIPPING CP ---
print("\n--- Skipping CP Decomposition ---")

# --- 2. Upsample Chunks (Multi-Stage - NO Amplitude Correction Needed Internally) ---
# (Upsampling code remains the same)
print("\n--- Upsampling chunks using MULTI-STAGE resample_poly ---")
fs_interim = fs_sdr * 5
print(f"Intermediate sample rate for resampling: {fs_interim / 1e6:.2f} MHz")
upsampled_chunks = []
debug_rms_upsample_input = []
debug_rms_upsample_output = []
for i, chunk_data in tqdm(enumerate(aligned_chunks), total=len(aligned_chunks), desc="Upsampling"):
    meta = loaded_metadata[i]; chunk_duration = len(chunk_data)/fs_sdr if fs_sdr > 0 else 0
    num_samples_chunk_recon = int(round(chunk_duration * fs_recon))
    if len(chunk_data) < 2: upsampled_chunks.append(np.zeros(num_samples_chunk_recon, dtype=complex)); debug_rms_upsample_input.append(0.0); debug_rms_upsample_output.append(0.0); continue
    try:
        rms_in = np.sqrt(np.mean(np.abs(chunk_data)**2)); debug_rms_upsample_input.append(rms_in)
        if rms_in < 1e-12: upsampled_chunks.append(np.zeros(num_samples_chunk_recon, dtype=complex)); debug_rms_upsample_output.append(0.0); continue
        up1=int(round(fs_interim)); down1=int(round(fs_sdr)); common1=np.gcd(up1,down1); up1//=common1; down1//=common1
        interim_chunk = sig.resample_poly(chunk_data, up1, down1)
        up2=int(round(fs_recon)); down2=int(round(fs_interim)); common2=np.gcd(up2,down2); up2//=common2; down2//=common2
        upsampled_chunk = sig.resample_poly(interim_chunk, up2, down2)
        if len(upsampled_chunk) > num_samples_chunk_recon: upsampled_chunk = upsampled_chunk[:num_samples_chunk_recon]
        elif len(upsampled_chunk) < num_samples_chunk_recon: upsampled_chunk = np.pad(upsampled_chunk, (0, num_samples_chunk_recon - len(upsampled_chunk)))
        upsampled_chunk = upsampled_chunk.astype(np.complex128); rms_out = np.sqrt(np.mean(np.abs(upsampled_chunk)**2)); debug_rms_upsample_output.append(rms_out)
        upsampled_chunks.append(upsampled_chunk.copy())
        if i == 0: # Debug plots for first chunk
             print("Plotting first upsampled chunk (MULTI-STAGE)..."); plt.figure(figsize=(12,4)); time_axis_debug = np.arange(len(upsampled_chunk))/fs_recon*1e6
             plt.plot(time_axis_debug, upsampled_chunk.real, label='Real'); plt.plot(time_axis_debug, upsampled_chunk.imag, label='Imag', alpha=0.7); plt.title('First Upsampled Chunk'); plt.xlabel('Time (µs)'); plt.ylabel('Amp'); plt.legend(); plt.grid(True)
             plot_limit_samples=min(len(upsampled_chunk),int(5*fs_recon*1e-6)); x_limit = time_axis_debug[plot_limit_samples] if plot_limit_samples>0 and plot_limit_samples<len(time_axis_debug) else 5; plt.xlim(-0.3, x_limit); plt.ylim(-EXPECTED_RMS*3, EXPECTED_RMS*3); plt.show()
             print("Plotting Multi-Stage Spectra Comparison..."); plt.figure(figsize=(12, 6))
             n=len(chunk_data); f=np.fft.fftshift(np.fft.fftfreq(n,d=1/fs_sdr)); s=np.fft.fftshift(np.fft.fft(chunk_data)); db=20*np.log10(np.abs(s)+1e-12); db-=np.max(db); plt.plot(f/1e6, db, label=f'BEFORE (Fs={fs_sdr/1e6:.1f}MHz)', alpha=0.8)
             n=len(upsampled_chunk); f=np.fft.fftshift(np.fft.fftfreq(n,d=1/fs_recon)); s=np.fft.fftshift(np.fft.fft(upsampled_chunk)); db=20*np.log10(np.abs(s)+1e-12); db-=np.max(db); plt.plot(f/1e6, db, label=f'AFTER (Fs={fs_recon/1e6:.1f}MHz)', ls='--', alpha=0.8)
             plt.title('Spectra Comparison Upsampling'); plt.xlabel('Freq (MHz rel Chunk Center)'); plt.ylabel('Mag (dB)'); plt.ylim(bottom=-120); plt.legend(); plt.grid(True); plt.show()
    except Exception as resample_e: print(f"Error resampling chunk {i}: {resample_e}. Appending zeros."); upsampled_chunks.append(np.zeros(num_samples_chunk_recon, dtype=complex)); debug_rms_upsample_input.append(0.0); debug_rms_upsample_output.append(0.0)
print("\n--- RMS Amplitudes Before/After Upsampling ---"); print("Chunk | RMS Before Upsample | RMS After Upsample"); print("------|---------------------|-------------------")
min_len_rms = min(len(debug_rms_upsample_input), len(debug_rms_upsample_output));
for i in range(min_len_rms): print(f"{i:<5d} | {debug_rms_upsample_input[i]:.4e}          | {debug_rms_upsample_output[i]:.4e}")


# --- 3. Stitching (DIRECT REPLACE TEST) ---
print("\n--- Stitching Upsampled Chunks (DIRECT REPLACE TEST) ---")
chunk_duration_s = len(loaded_chunks[0]) / fs_sdr if fs_sdr > 0 else 0
time_advance_per_chunk = chunk_duration_s * (1.0 - overlap_factor) + tuning_delay
total_duration_recon = chunk_duration_s + (len(loaded_chunks) - 1) * time_advance_per_chunk
num_samples_recon = int(round(total_duration_recon * fs_recon))

reconstructed_signal = np.zeros(num_samples_recon, dtype=complex)
# sum_of_windows = np.zeros(num_samples_recon, dtype=float) # Not needed for direct replace

print(f"Reconstruction target: {num_samples_recon} samples @ {fs_recon/1e6:.2f} MHz (Duration: {total_duration_recon*1e6:.1f} us)")

current_recon_time_start = 0.0
plt.figure(figsize=(14, 8)); num_subplot_rows = int(np.ceil(len(upsampled_chunks)/2)); debug_plot_len=1000

for i, up_chunk in tqdm(enumerate(upsampled_chunks), total=len(upsampled_chunks), desc="Direct Replace Stitching"):
    meta = loaded_metadata[i]
    if len(up_chunk) == 0:
        if i < len(loaded_chunks) - 1: current_recon_time_start += time_advance_per_chunk
        continue

    rms_in_stitching = np.sqrt(np.mean(np.abs(up_chunk)**2))
    print(f"  Placing Chunk {i}: RMS = {rms_in_stitching:.4e}")

    start_idx_recon = int(round(current_recon_time_start * fs_recon))
    num_samples_in_chunk = len(up_chunk)
    end_idx_recon = min(start_idx_recon + num_samples_in_chunk, num_samples_recon)
    actual_len = end_idx_recon - start_idx_recon
    if actual_len <= 0:
        if i < len(loaded_chunks) - 1: current_recon_time_start += time_advance_per_chunk
        continue

    # --- Perform the DIRECT REPLACE ---
    try:
        segment_to_place = up_chunk[:actual_len] # Get the segment
        if not np.all(np.isfinite(segment_to_place)): print(f"*** WARNING: Non-finite segment_to_place chunk {i} ***"); segment_to_place[~np.isfinite(segment_to_place)] = 0
        reconstructed_signal[start_idx_recon:end_idx_recon] = segment_to_place # Overwrite section
        if not np.all(np.isfinite(reconstructed_signal[start_idx_recon:end_idx_recon])): print(f"*** WARNING: Non-finite recon after placing chunk {i} ***")
    except Exception as place_e: print(f"Error during direct place chunk {i}: {place_e}"); continue
    # --- END DIRECT REPLACE ---

    # --- DEBUG Plotting ---
    if i < num_subplot_rows * 2:
        ax = plt.subplot(num_subplot_rows, 2, i + 1)
        plot_end_idx = end_idx_recon + int(0.1*num_samples_in_chunk); plot_end_idx = min(plot_end_idx, num_samples_recon)
        plot_indices = min(plot_end_idx, start_idx_recon + debug_plot_len); # Limit points from start of placement
        plot_indices = min(plot_indices, num_samples_recon) # Ensure within bounds
        time_axis = np.arange(plot_indices) / fs_recon * 1e6 # Only plot up to plot_indices
        if plot_indices > 0 :
            ax.plot(time_axis, reconstructed_signal[:plot_indices].real, label=f'Real (After Place {i})')
            ax.plot(time_axis, reconstructed_signal[:plot_indices].imag, label=f'Imag (After Place {i})', alpha=0.7)
            ax.set_title(f'Recon State After Placing Chunk {i}', fontsize='small'); ax.grid(True)
            max_val_plot = np.nanmax(np.abs(reconstructed_signal[:plot_indices])) if plot_indices > 0 else EXPECTED_RMS
            if max_val_plot < 1e-12 or not np.isfinite(max_val_plot): max_val_plot = EXPECTED_RMS
            ax.set_ylim(-max_val_plot*1.5, max_val_plot*1.5) # Dynamic zoom
        else: ax.set_title(f'Chunk {i} - No data', fontsize='small')
    # --- END DEBUG ---

    # CORRECT Time Advancement
    if i < len(loaded_chunks) - 1: current_recon_time_start += time_advance_per_chunk

plt.suptitle("Progressive Reconstruction Signal (Direct Replace)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()
print("\nDirect Replace Stitching loop complete.")

# --- SKIPPING Normalization Step ---
print("\n--- SKIPPING Normalization Step (Not Applicable for Direct Replace Test) ---")
# The previous normalization relied on sum_of_windows, which we didn't calculate

# --- 4. Evaluation & Visualization ---
print("\n--- Evaluating Reconstruction ---")
print("Regenerating ground truth baseband for comparison...")
gt_duration = total_duration_recon
num_samples_gt_compare = int(round(gt_duration * fs_recon))
t_gt_compare = np.linspace(0, gt_duration, num_samples_gt_compare, endpoint=False)
gt_baseband_compare = np.zeros(num_samples_gt_compare, dtype=complex)
mod = global_attrs.get('modulation', 'qam16'); bw_gt = global_attrs['total_signal_bandwidth_hz']
if mod.lower() == 'qam16':
    symbol_rate_gt = bw_gt / 4; num_symbols_gt = int(np.ceil(gt_duration * symbol_rate_gt))
    symbols = (np.random.choice([-3,-1,1,3], size=num_symbols_gt)+1j*np.random.choice([-3,-1,1,3], size=num_symbols_gt))/np.sqrt(10)
    samples_per_symbol_gt=max(1, int(round(fs_recon/symbol_rate_gt))); baseband_symbols = np.repeat(symbols, samples_per_symbol_gt)
    len_to_take = min(len(baseband_symbols), num_samples_gt_compare); gt_baseband_compare[:len_to_take] = baseband_symbols[:len_to_take]
else: print(f"Warning: GT regen not implemented for {mod}")

# Evaluation without normalization (Direct comparison of power)
# Align amplitudes for plotting and fair MSE comparison
mean_power_gt_eval = np.mean(np.abs(gt_baseband_compare)**2)
mean_power_recon_eval = np.mean(np.abs(reconstructed_signal)**2) # Use the raw direct-replace signal

print(f"RMS GT = {np.sqrt(mean_power_gt_eval):.4e}, RMS Recon = {np.sqrt(mean_power_recon_eval):.4e}")

if mean_power_recon_eval > 1e-20 and np.isfinite(mean_power_recon_eval) and mean_power_gt_eval > 1e-20:
    # Scale reconstructed signal to match GT power for evaluation
    power_scale = np.sqrt(mean_power_gt_eval / mean_power_recon_eval)
    reconstructed_signal_aligned = reconstructed_signal * power_scale
    print(f"Applied alignment scale factor: {power_scale:.4f}")
else:
    print("Warning: Cannot align power for MSE calculation.")
    reconstructed_signal_aligned = reconstructed_signal # Plot unaligned

if len(gt_baseband_compare) != len(reconstructed_signal_aligned):
     print(f"ERROR: Length mismatch before MSE! GT={len(gt_baseband_compare)}, Recon={len(reconstructed_signal_aligned)}")
     mse = np.inf
elif not np.all(np.isfinite(reconstructed_signal_aligned)):
    print("Error: Non-finite recon signal after alignment")
    mse = np.inf
else:
     # MSE calculation comparing power-aligned signals
     mse = np.mean(np.abs(gt_baseband_compare - reconstructed_signal_aligned)**2)
print(f"MSE (Direct Replace, Power Aligned): {mse:.4e}")


# Plotting
plt.style.use('seaborn-v0_8-darkgrid'); fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
mean_power_gt_plot = np.mean(np.abs(gt_baseband_compare)**2); gt_plot_norm = gt_baseband_compare / np.sqrt(mean_power_gt_plot) if mean_power_gt_plot > 1e-20 else gt_baseband_compare
plot_samples = min(plot_length, len(t_gt_compare), len(gt_plot_norm)); time_axis_plot = t_gt_compare[:plot_samples] * 1e6
axs[0].plot(time_axis_plot, gt_plot_norm[:plot_samples].real, label='GT (Real)'); axs[0].plot(time_axis_plot, gt_plot_norm[:plot_samples].imag, label='GT (Imag)', alpha=0.7); axs[0].set_title('Ground Truth (Normalized)'); axs[0].set_ylabel('Amp'); axs[0].legend(fontsize='small'); axs[0].grid(True)
# Align the plotting data based on its own power vs GT plot power
mean_power_recon_plot = np.mean(np.abs(reconstructed_signal_aligned)**2); recon_plot_norm = reconstructed_signal_aligned / np.sqrt(mean_power_recon_plot) if mean_power_recon_plot > 1e-20 else reconstructed_signal_aligned
plot_samples_recon = min(plot_length, len(recon_plot_norm)); plot_data_recon = recon_plot_norm[:plot_samples_recon]; plot_data_recon[~np.isfinite(plot_data_recon)] = 0
axs[1].plot(time_axis_plot[:plot_samples_recon], plot_data_recon.real, label='Recon (Real)'); axs[1].plot(time_axis_plot[:plot_samples_recon], plot_data_recon.imag, label='Recon (Imag)', alpha=0.7); axs[1].set_title(f'Reconstructed (Direct Replace, MSE: {mse:.2e})'); axs[1].set_ylabel('Amp'); axs[1].legend(fontsize='small'); axs[1].grid(True)
plot_samples_error = min(plot_samples, plot_samples_recon); error_signal = gt_plot_norm[:plot_samples_error] - plot_data_recon[:plot_samples_error]
axs[2].plot(time_axis_plot[:plot_samples_error], error_signal.real, label='Error (Real)'); axs[2].plot(time_axis_plot[:plot_samples_error], error_signal.imag, label='Error (Imag)', alpha=0.7); axs[2].set_title('Error'); axs[2].set_xlabel('Time (µs)'); axs[2].set_ylabel('Amp'); axs[2].legend(fontsize='small'); axs[2].grid(True)
plt.tight_layout(); plt.show()
plt.figure(figsize=(12, 7))
if len(recon_plot_norm)>1 and np.all(np.isfinite(recon_plot_norm)): f_recon=np.fft.fftshift(np.fft.fftfreq(len(recon_plot_norm),d=1/fs_recon)); spec_recon = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(recon_plot_norm)))+1e-12); spec_recon-=np.nanmax(spec_recon); plt.plot(f_recon/1e6, spec_recon, label='Recon Spec', ls='--', alpha=0.8)
else: print("Skipping recon spectrum plot.")
if len(gt_plot_norm)>1: f_gt=np.fft.fftshift(np.fft.fftfreq(len(gt_plot_norm),d=1/fs_recon)); spec_gt = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(gt_plot_norm)))+1e-12); spec_gt-=np.nanmax(spec_gt); plt.plot(f_gt/1e6, spec_gt, label='GT Spec', alpha=0.8)
else: print("Skipping GT spectrum plot.")
plt.title('Spectra Comparison'); plt.xlabel('Freq (MHz)'); plt.ylabel('Mag (dB)'); plt.ylim(bottom=spectrum_ylim); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

print("\nSimplified script finished.")