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

stitching_window_type = 'blackmanharris'  # Changed from 'hann' to blackmanharris for better spectral properties
plot_length = 5000
spectrum_ylim = -100 # Adjust if needed based on Power Spectrum results
EXPECTED_RMS = 1.39e-02 # Target RMS after initial scaling

# --- WPD Phase Correction Parameters ---
wpd_wavelet = 'db4'  # Wavelet for WPD
wpd_level = 4      # Decomposition level for WPD (adjust based on needs/computation)
# Note: Higher levels give finer frequency but shorter coefficient sequences
# NOTE: This WPD implementation focuses on intra-chunk phase detrending.

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
            group_name = f'chunk_{i:03d}' # Construct group name for loading
            if group_name in f:
                group = f[group_name]
                chunk_data = group['iq_data'][:].astype(np.complex128)
                meta = {key: value for key, value in group.attrs.items()} # Load full meta again
                loaded_chunks.append(chunk_data)
                loaded_metadata.append(meta) # Store full metadata dictionary
            else:
                print(f"Warning: Chunk group '{group_name}' not found during data loading. Skipping.")
                # Append placeholders if needed, or handle downstream logic carefully
                # loaded_chunks.append(np.array([], dtype=np.complex128)) # Example placeholder
                # loaded_metadata.append({}) # Example placeholder


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
    max_abs_after = np.max(np.abs(scaled_chunk)) if len(scaled_chunk)>0 else 0
    # Relax tolerance slightly for RMS check after scaling
    if not np.isclose(rms_after_scaling, EXPECTED_RMS, rtol=5e-3):
        print(f"WARNING: Chunk {i} RMS scaling mismatch ({rms_after_scaling:.4e} vs {EXPECTED_RMS:.4e})")
    print(f"{i:<5d} | {rms_before_scaling:.4e} | {scaling_factor:<14.4f} | {rms_after_scaling:.4e} | {max_abs_after:.4e}")
    scaled_loaded_chunks.append(scaled_chunk)
if not scaling_successful: print("\nERROR: Non-finite values detected during scaling. Cannot proceed."); sys.exit(1)
print("--- Initial Amplitude Scaling Complete ---")


# --- Skip Adaptive Filtering Pre-Alignment ---
print("\n--- SKIPPING Adaptive Filtering Pre-Alignment (FOR DEBUGGING) ---")
aligned_chunks = scaled_loaded_chunks # Use scaled chunks directly

# --- 2. IMPROVED: Upsampling Chunks using Polyphase Filters ---
print("\n--- Upsampling chunks (using polyphase filter) ---")
upsampled_chunks = []
debug_rms_upsample_input = [] # Keep debug lists if needed
debug_rms_upsample_output = []
debug_max_abs_upsample_input = []
debug_max_abs_upsample_output = []

for i, chunk_data in tqdm(enumerate(aligned_chunks), total=len(aligned_chunks), desc="Upsampling"):
    chunk_duration = len(chunk_data)/GLBL_SDR_SAMPLE_RATE if GLBL_SDR_SAMPLE_RATE > 0 else 0
    num_samples_chunk_recon = int(round(chunk_duration * GLBL_RECON_SAMPLE_RATE))

    if len(chunk_data) < 2:
        upsampled_chunks.append(np.zeros(num_samples_chunk_recon, dtype=complex))
        continue

    try:
        rms_in = np.sqrt(np.mean(np.abs(chunk_data)**2))
        max_abs_in = np.max(np.abs(chunk_data)) if len(chunk_data)>0 else 0
        debug_rms_upsample_input.append(rms_in)
        debug_max_abs_upsample_input.append(max_abs_in)

        if rms_in < 1e-12:
            upsampled_chunks.append(np.zeros(num_samples_chunk_recon, dtype=complex))
            continue

        # Calculate resampling ratio - ensure integer factors for resample_poly
        common_divisor = np.gcd(int(GLBL_RECON_SAMPLE_RATE), int(GLBL_SDR_SAMPLE_RATE))
        up_factor = int(GLBL_RECON_SAMPLE_RATE // common_divisor)
        down_factor = int(GLBL_SDR_SAMPLE_RATE // common_divisor)
        # print(f"Chunk {i}: Resampling {len(chunk_data)} samples. Up: {up_factor}, Down: {down_factor}. Target: {num_samples_chunk_recon}") # Verbose debug

        # Use polyphase resampling for better phase preservation
        resampled_real = sig.resample_poly(chunk_data.real, up=up_factor, down=down_factor, window=('kaiser', 5.0))
        resampled_imag = sig.resample_poly(chunk_data.imag, up=up_factor, down=down_factor, window=('kaiser', 5.0))

        # Combine back to complex
        upsampled_chunk = resampled_real + 1j * resampled_imag

        # Trim or pad to exact length needed
        current_len = len(upsampled_chunk)
        if current_len > num_samples_chunk_recon:
            upsampled_chunk = upsampled_chunk[:num_samples_chunk_recon]
            # print(f"  Trimmed {current_len} to {num_samples_chunk_recon}") # Verbose debug
        elif current_len < num_samples_chunk_recon:
            pad_length = num_samples_chunk_recon - current_len
            upsampled_chunk = np.pad(upsampled_chunk, (0, pad_length), mode='constant')
            # print(f"  Padded {current_len} to {num_samples_chunk_recon}") # Verbose debug

        upsampled_chunk = upsampled_chunk.astype(np.complex128)

        # Ensure amplitude scaling is maintained (resample_poly preserves power approximately)
        current_rms = np.sqrt(np.mean(np.abs(upsampled_chunk)**2))
        if current_rms > 1e-12 and rms_in > 1e-12: # Avoid scaling zero-power signals
             scale_correction = rms_in / current_rms
             upsampled_chunk *= scale_correction

        rms_out = np.sqrt(np.mean(np.abs(upsampled_chunk)**2))
        max_abs_out = np.max(np.abs(upsampled_chunk)) if len(upsampled_chunk)>0 else 0
        debug_rms_upsample_output.append(rms_out)
        debug_max_abs_upsample_output.append(max_abs_out)
        upsampled_chunks.append(upsampled_chunk.copy())

        # --- Plotting first chunk ---
        if i == 0:
             print("Plotting first upsampled chunk (polyphase method)...")
             plt.figure(figsize=(12,4))
             time_axis_debug = np.arange(len(upsampled_chunk))/GLBL_RECON_SAMPLE_RATE*1e6
             plt.plot(time_axis_debug, upsampled_chunk.real, label='Real')
             plt.plot(time_axis_debug, upsampled_chunk.imag, label='Imag', alpha=0.7)
             plt.title(f'First Upsampled Chunk (Polyphase, RMS={rms_out:.3e})')
             plt.xlabel('Time (µs)')
             plt.ylabel('Amp')
             plt.legend()
             plt.grid(True)
             ylim_abs = max(max_abs_out * 1.2 if np.isfinite(max_abs_out) else 0.1, 0.05)
             plt.ylim(-ylim_abs, ylim_abs)
             plt.xlim(-0.1, 5.0) # Adjust xlim if needed
             plt.show(block=False) # Show non-blocking
             plt.pause(0.1) # Allow plot to render

    except Exception as resample_e:
        print(f"Error resampling chunk {i}: {resample_e}. Appending zeros.")
        upsampled_chunks.append(np.zeros(num_samples_chunk_recon, dtype=complex))

# --- 3. IMPROVED: Enhanced Phase Correlation and Correction ---
print("\n--- Performing Improved Phase Correlation ---")

# Calculate overlap samples with increased overlap factor for better stitching
# Check if loaded_chunks has data before accessing element 0
chunk_duration_s = 0
if len(loaded_chunks) > 0 and GLBL_SDR_SAMPLE_RATE is not None and GLBL_SDR_SAMPLE_RATE > 0:
    chunk_duration_s = len(loaded_chunks[0]) / GLBL_SDR_SAMPLE_RATE
elif len(upsampled_chunks) > 0 and GLBL_RECON_SAMPLE_RATE is not None and GLBL_RECON_SAMPLE_RATE > 0:
    # Fallback: estimate duration from first upsampled chunk if loaded_chunks is empty/invalid
    chunk_duration_s = len(upsampled_chunks[0]) / GLBL_RECON_SAMPLE_RATE
    print("Warning: Estimating chunk duration from first upsampled chunk.")
else:
    print("Warning: Cannot determine chunk duration for overlap calculation. Check sample rates and chunk data.")

effective_overlap = max(0.25, GLBL_OVERLAP_FACTOR)  # Ensure minimum 25% overlap
overlap_samples_recon = int(round(chunk_duration_s * effective_overlap * GLBL_RECON_SAMPLE_RATE)) if GLBL_RECON_SAMPLE_RATE is not None else 0
print(f"Enhanced Overlap samples for Correlation: {overlap_samples_recon}")

# Initialize
improved_phase_chunks = []
estimated_cumulative_phases_improved = [0.0] # Track estimated phases

if len(upsampled_chunks) > 0:
    improved_phase_chunks.append(upsampled_chunks[0]) # First chunk is reference

    for i in tqdm(range(1, len(upsampled_chunks)), desc="Improved Phase Correlation"):
        # Check if overlap is meaningful and segments are long enough
        if overlap_samples_recon <= 0 or len(upsampled_chunks[i-1]) < overlap_samples_recon or len(upsampled_chunks[i]) < overlap_samples_recon:
            status = f"overlap samples={overlap_samples_recon}" if overlap_samples_recon <= 0 else f"chunk lengths prev={len(upsampled_chunks[i-1])}, curr={len(upsampled_chunks[i])}"
            print(f"Warning: Insufficient overlap ({status}) for chunk {i}. Skipping correlation, applying previous phase correction.")
            estimated_cumulative_phases_improved.append(estimated_cumulative_phases_improved[-1])
            # Apply the *previous* cumulative phase correction to the current chunk
            corrected_chunk = upsampled_chunks[i] * np.exp(-1j * estimated_cumulative_phases_improved[-1])
            improved_phase_chunks.append(corrected_chunk)
            continue

        # Extract overlapping segments
        prev_segment = upsampled_chunks[i-1][-overlap_samples_recon:]
        curr_segment = upsampled_chunks[i][:overlap_samples_recon]

        # Compute cross-correlation of magnitudes to find optimal lag
        max_lag = min(100, overlap_samples_recon // 10) # Limit lag search range

        xcorr_mag = np.correlate(np.abs(curr_segment), np.abs(prev_segment), mode='full')
        n_curr = len(curr_segment)
        n_prev = len(prev_segment)
        lags = np.arange(-(n_curr - 1), n_prev)

        # Find the peak within the allowed lag range
        search_indices = np.where(np.abs(lags) <= max_lag)[0]

        delta_phi = 0.0 # Default phase difference

        if len(search_indices) > 0:
            valid_search_indices = search_indices[(search_indices >= 0) & (search_indices < len(xcorr_mag))]
            if len(valid_search_indices) > 0:
                best_lag_idx_in_search = np.argmax(np.abs(xcorr_mag[valid_search_indices]))
                best_overall_idx = valid_search_indices[best_lag_idx_in_search]
                best_lag = lags[best_overall_idx]
                # print(f"  Chunk {i}: Best lag = {best_lag}") # Debug print

                # Adjust segments based on the determined lag for phase estimation
                if best_lag > 0:
                    aligned_curr = curr_segment[best_lag:]
                    aligned_prev = prev_segment[:len(aligned_curr)]
                elif best_lag < 0:
                    aligned_prev = prev_segment[abs(best_lag):]
                    aligned_curr = curr_segment[:len(aligned_prev)]
                else:
                    aligned_curr = curr_segment
                    aligned_prev = prev_segment

                # Estimate phase with better aligned signals
                if len(aligned_curr) > 0 and len(aligned_prev) > 0:
                    min_len = min(len(aligned_curr), len(aligned_prev))
                    sum_corr = np.sum(aligned_curr[:min_len] * np.conj(aligned_prev[:min_len]))
                    if np.abs(sum_corr) > 1e-12:
                        delta_phi = np.angle(sum_corr)
                    else:
                        delta_phi = 0.0
                else:
                     delta_phi = 0.0
            else:
                # Fallback to simple correlation if lag search fails
                sum_corr = np.sum(curr_segment * np.conj(prev_segment))
                if np.abs(sum_corr) > 1e-12: delta_phi = np.angle(sum_corr)
                else: delta_phi = 0.0

        else:
             # Fallback to simple correlation if lag search fails or max_lag=0
            sum_corr = np.sum(curr_segment * np.conj(prev_segment))
            if np.abs(sum_corr) > 1e-12: delta_phi = np.angle(sum_corr)
            else: delta_phi = 0.0

        # Update cumulative phase
        prev_cumulative = estimated_cumulative_phases_improved[-1]
        current_cumulative_phase = prev_cumulative + delta_phi
        estimated_cumulative_phases_improved.append(current_cumulative_phase)

        # Apply phase correction
        corrected_chunk = upsampled_chunks[i] * np.exp(-1j * current_cumulative_phase)
        improved_phase_chunks.append(corrected_chunk)
else:
    print("No upsampled chunks available to perform phase correlation.")

print("--- Improved Phase Correlation Complete ---")


# --- 4. IMPLEMENT: Wavelet Packet Decomposition (WPD) for Phase Correction ---
# **** CORRECTED WPD SECTION ****
print("\n--- Performing Wavelet-Based Phase Correction (Intra-Chunk Detrending) ---")
wpd_phase_corrected_chunks = []
wavelet_obj = None
try:
    wavelet_obj = pywt.Wavelet(wpd_wavelet)
except Exception as e:
    print(f"Fatal Error: Could not initialize wavelet '{wpd_wavelet}': {e}. Skipping WPD.")
    wpd_phase_corrected_chunks = improved_phase_chunks # Pass through if wavelet fails

if wavelet_obj: # Proceed only if wavelet was initialized
    for i, chunk in tqdm(enumerate(improved_phase_chunks), total=len(improved_phase_chunks), desc="WPD Processing"):
        # Skip if chunk is too short for WPD or empty
        actual_max_level = 0
        if len(chunk) > 0:
            try:
                actual_max_level = pywt.dwt_max_level(len(chunk), wavelet_obj)
            except Exception as max_level_e:
                print(f"Warning: Could not determine max WPD level for chunk {i} (len={len(chunk)}): {max_level_e}. Skipping WPD.")
                wpd_phase_corrected_chunks.append(chunk)
                continue

        if len(chunk) <= 2 or wpd_level > actual_max_level or wpd_level <= 0:
            if len(chunk) > 2 and wpd_level > 0:
                print(f"  Warning: Chunk {i} length ({len(chunk)}) insufficient for WPD level {wpd_level} (max is {actual_max_level}). Skipping WPD.")
            elif wpd_level <= 0:
                print(f"  Warning: Invalid WPD level ({wpd_level}). Skipping WPD.")
            wpd_phase_corrected_chunks.append(chunk)
            continue

        try:
            # Process real and imaginary parts
            wp_real = pywt.WaveletPacket(data=np.real(chunk), wavelet=wpd_wavelet, mode='symmetric', maxlevel=wpd_level)
            wp_imag = pywt.WaveletPacket(data=np.imag(chunk), wavelet=wpd_wavelet, mode='symmetric', maxlevel=wpd_level)

            # Get node objects at the specified level
            nodes_real = wp_real.get_level(wpd_level, 'natural')

            # Iterate through node objects from the real tree
            for node_real in nodes_real:
                try:
                    current_node_path = node_real.path # Get path string from the node object
                    if not isinstance(current_node_path, str): # Explicit check
                        print(f"  Warning: Node path '{current_node_path}' (type: {type(current_node_path)}) is not a string for chunk {i}. Skipping node.")
                        continue

                    # Check if corresponding path exists in the imaginary tree
                    if current_node_path not in wp_imag:
                        print(f"  Warning: Node path {current_node_path} not found in imaginary WPD tree for chunk {i}. Skipping correction for this node.")
                        continue
                    node_imag = wp_imag[current_node_path] # Access imag node using the path string

                    # Access data directly from node objects
                    real_data = np.array(node_real.data)
                    imag_data = np.array(node_imag.data)
                    coeff_complex = real_data + 1j * imag_data

                    # Perform phase detrending if enough data points exist
                    if len(coeff_complex) > 1:
                        inst_phase = np.unwrap(np.angle(coeff_complex + 1e-30)) # Add epsilon
                        x = np.arange(len(inst_phase))
                        try:
                            A = np.vstack([x, np.ones(len(x))]).T
                            phase_slope, phase_intercept = np.linalg.lstsq(A, inst_phase, rcond=None)[0]
                            linear_phase_trend = phase_intercept + phase_slope * x
                            phase_correction_factor = np.exp(-1j * linear_phase_trend)
                            corrected_coeffs = coeff_complex * phase_correction_factor

                            # Update data directly on the node objects
                            node_real.data = corrected_coeffs.real
                            node_imag.data = corrected_coeffs.imag
                        except np.linalg.LinAlgError as e:
                            print(f"  Warning: Phase trend calculation failed (LinAlgError) for node {current_node_path} in chunk {i}: {e}. Skipping node.")
                        except Exception as e_phase:
                            print(f"  Warning: Unexpected error during phase trend for node {current_node_path} in chunk {i}: {e_phase}. Skipping node.")

                except KeyError:
                     # Should be caught by 'if current_node_path not in wp_imag' check
                     print(f"  Internal Warning: KeyError accessing node {current_node_path} in chunk {i}.")
                except Exception as node_e:
                     path_info = getattr(node_real, 'path', 'unknown') # Safely get path for error msg
                     print(f"  Warning: Error processing node '{path_info}' in chunk {i}: {node_e}. Skipping node correction.")


            # Reconstruct signal - use update=False as data was modified directly
            corrected_real = wp_real.reconstruct(update=False)
            corrected_imag = wp_imag.reconstruct(update=False)

            # Combine complex signal and ensure proper length
            corrected_chunk = corrected_real + 1j * corrected_imag
            if len(corrected_chunk) != len(chunk):
                 print(f"  Info: WPD reconstruction changed length chunk {i} from {len(chunk)} to {len(corrected_chunk)}. Adjusting.")
                 corrected_chunk = corrected_chunk[:len(chunk)] # Trim
                 if len(corrected_chunk) < len(chunk): # Pad if too short
                      pad_len = len(chunk) - len(corrected_chunk)
                      corrected_chunk = np.pad(corrected_chunk, (0, pad_len), 'constant')

            # Normalize RMS to maintain consistency
            chunk_rms = np.sqrt(np.mean(np.abs(chunk)**2))
            corrected_rms = np.sqrt(np.mean(np.abs(corrected_chunk)**2))
            if chunk_rms > 1e-12 and corrected_rms > 1e-12:
                corrected_chunk *= (chunk_rms / corrected_rms)

            wpd_phase_corrected_chunks.append(corrected_chunk)

        except ValueError as ve:
            print(f"  Error during WPD setup/decomposition for chunk {i} (ValueError): {ve}")
            wpd_phase_corrected_chunks.append(chunk) # Use original on failure
        except Exception as e:
            print(f"  General Error in WPD processing chunk {i}: {e}")
            wpd_phase_corrected_chunks.append(chunk) # Use original on failure
else:
    # This case handles when wavelet_obj failed initialization
    pass

print("--- WPD Phase Correction (Intra-Chunk Detrending) Complete ---")


# --- 5. Normalize Each Chunk Before Stitching ---
print("\n--- Normalizing chunks before stitching ---")
normalized_chunks = []
print("Chunk | RMS Before Norm | RMS After Norm")
print("------|-----------------|---------------")
for i, chunk in enumerate(wpd_phase_corrected_chunks):
    rms_before = np.nan
    rms_after = np.nan
    norm_chunk = chunk # Default to original chunk if checks fail

    if len(chunk) > 0 and np.all(np.isfinite(chunk)):
        chunk_rms = np.sqrt(np.mean(np.abs(chunk)**2))
        rms_before = chunk_rms
        if chunk_rms > 1e-12:
            scale = EXPECTED_RMS / chunk_rms
            norm_chunk = chunk * scale
            rms_after = np.sqrt(np.mean(np.abs(norm_chunk)**2))
        else:
            norm_chunk = np.zeros_like(chunk)
            rms_after = 0.0
    elif len(chunk) == 0:
         norm_chunk = chunk
         rms_before = 0.0
         rms_after = 0.0
    else:
         print(f"Warning: Chunk {i} contains non-finite values before normalization. Replacing with zeros.")
         norm_chunk = np.zeros_like(chunk)
         rms_before = np.nan
         rms_after = 0.0

    normalized_chunks.append(norm_chunk)
    print(f"{i:<5d} | {rms_before:<15.4e} | {rms_after:.4e}")

print("--- Normalization Complete ---")


# --- 6. Stitching (Overlap-Add) with Improved Window Functions ---
print("\n--- Performing Stitching with Improved Windows ---")
time_advance_per_chunk = chunk_duration_s * (1.0 - effective_overlap)
total_duration_recon = chunk_duration_s + (max(0, len(loaded_chunks) - 1)) * (time_advance_per_chunk + GLBL_TUNING_DELAY) # Use max(0,...) for safety
num_samples_recon = int(round(total_duration_recon * GLBL_RECON_SAMPLE_RATE)) if GLBL_RECON_SAMPLE_RATE is not None else 0

if num_samples_recon <= 0:
    print(f"Error: Invalid calculated reconstruction samples ({num_samples_recon}). Check rates, duration, overlap. Exiting.")
    sys.exit(1)

reconstructed_signal = np.zeros(num_samples_recon, dtype=complex)
sum_of_windows = np.zeros(num_samples_recon, dtype=float)

print(f"Reconstruction target buffer: {num_samples_recon} samples @ {GLBL_RECON_SAMPLE_RATE/1e6:.2f} MHz (Est. Duration: {total_duration_recon*1e6:.1f} us)")
print(f"Using stitching window: '{stitching_window_type}' with effective overlap factor: {effective_overlap:.2f}")

current_recon_time_start = 0.0

for i, chunk_to_add in tqdm(enumerate(normalized_chunks), total=len(normalized_chunks), desc="Stitching"):
    start_idx_recon = int(round(current_recon_time_start * GLBL_RECON_SAMPLE_RATE))
    num_samples_in_chunk = len(chunk_to_add)
    end_idx_recon = min(start_idx_recon + num_samples_in_chunk, num_samples_recon)
    actual_len = end_idx_recon - start_idx_recon

    if actual_len <= 0 or num_samples_in_chunk == 0:
        if i < len(normalized_chunks) - 1:
             current_recon_time_start += time_advance_per_chunk + GLBL_TUNING_DELAY
        continue

    window = np.ones(actual_len)
    if actual_len >= 2:
        try:
            window = sig.get_window(stitching_window_type, actual_len)
        except ValueError as e:
             print(f"Warning: Could not get window '{stitching_window_type}' for length {actual_len}. Using rectangular. Error: {e}")
             window = np.ones(actual_len)

    window_rms = np.sqrt(np.mean(window**2))
    if window_rms > 1e-9:
         normalized_window_for_sum = window / window_rms
    else:
         normalized_window_for_sum = np.ones_like(window)

    try:
        segment_data = chunk_to_add[:actual_len]
        if not np.all(np.isfinite(segment_data)):
            print(f"Warning: Non-finite values in chunk {i} segment BEFORE windowing. Zeroing.")
            segment_data = np.nan_to_num(segment_data)
        windowed_segment = segment_data * window
        reconstructed_signal[start_idx_recon:end_idx_recon] += windowed_segment
        sum_of_windows[start_idx_recon:end_idx_recon] += normalized_window_for_sum
        if not np.all(np.isfinite(reconstructed_signal[start_idx_recon:end_idx_recon])):
            print(f"Warning: Non-finite values after adding chunk {i}. Zeroing affected segment.")
            reconstructed_signal[start_idx_recon:end_idx_recon] = np.nan_to_num(reconstructed_signal[start_idx_recon:end_idx_recon])
    except ValueError as ve:
         print(f"Error overlap-add chunk {i} (ValueError): {ve}.")
    except Exception as add_e:
         print(f"Error overlap-add chunk {i}: {add_e}")

    if i < len(normalized_chunks) - 1:
        current_recon_time_start += time_advance_per_chunk + GLBL_TUNING_DELAY

print("\nStitching loop complete.")


# --- Normalization Step (Post-Stitching using Sum of Windows) ---
print("\n--- Normalizing reconstructed signal using Sum-of-Windows ---")
rms_before_norm = np.nan
max_abs_before_norm = np.nan
if len(reconstructed_signal) > 0:
    finite_mask = np.isfinite(reconstructed_signal)
    if np.any(finite_mask):
        rms_before_norm = np.sqrt(np.mean(np.abs(reconstructed_signal[finite_mask])**2))
        max_abs_before_norm = np.max(np.abs(reconstructed_signal[finite_mask]))
    if not np.all(finite_mask):
        print("Warning: Non-finite values detected in raw stitched signal before normalization.")

print("Signal Stats BEFORE Sum-of-Windows Normalization:")
print(f"  RMS (finite parts): {rms_before_norm:.4e}")
print(f"  Max Abs (finite parts): {max_abs_before_norm:.4e}")
print(f"  Sum of windows stats: Min={np.min(sum_of_windows):.4f}, Max={np.max(sum_of_windows):.4f}, Mean={np.mean(sum_of_windows):.4f}, Median={np.median(sum_of_windows):.4f}")

reliable_threshold = 1e-6
reliable_indices = np.where(sum_of_windows >= reliable_threshold)[0]
fallback_sum = 1.0

if len(reliable_indices) > 0:
    median_sum = np.median(sum_of_windows[reliable_indices])
    if np.isfinite(median_sum) and median_sum > 1e-9:
        fallback_sum = median_sum
    else:
        mean_sum_reliable = np.mean(sum_of_windows[reliable_indices])
        if np.isfinite(mean_sum_reliable) and mean_sum_reliable > 1e-9:
             fallback_sum = mean_sum_reliable
             print(f"Warning: Median sum ({median_sum}) not reliable. Using mean ({fallback_sum:.4f}) as fallback.")
        else:
             print(f"Warning: Median and Mean of reliable sum_of_windows are not reliable. Using default fallback {fallback_sum}.")
else:
    print(f"Warning: No reliable indices found for sum_of_windows. Using default fallback {fallback_sum}.")

print(f"Using fallback divisor value: {fallback_sum:.4f} for unreliable regions.")

sum_of_windows_divisor = sum_of_windows.copy()
unreliable_mask = (sum_of_windows < reliable_threshold) | (~np.isfinite(sum_of_windows))
sum_of_windows_divisor[unreliable_mask] = fallback_sum
sum_of_windows_divisor[np.abs(sum_of_windows_divisor) < 1e-15] = fallback_sum

reconstructed_signal_normalized = np.zeros_like(reconstructed_signal)
valid_divisor = np.isfinite(sum_of_windows_divisor) & (np.abs(sum_of_windows_divisor) > 1e-15)
signal_to_divide = np.nan_to_num(reconstructed_signal)

np.divide(signal_to_divide, sum_of_windows_divisor, out=reconstructed_signal_normalized, where=valid_divisor)

if not np.all(np.isfinite(reconstructed_signal_normalized)):
     print("Warning: Non-finite values detected AFTER robust normalization. Zeroing.")
     reconstructed_signal_normalized = np.nan_to_num(reconstructed_signal_normalized)

# --- Final RMS Scaling ---
print("\n--- Applying Final RMS Scaling ---")
rms_after_sow_norm = 0.0
if len(reconstructed_signal_normalized) > 0:
     rms_after_sow_norm = np.sqrt(np.mean(np.abs(reconstructed_signal_normalized)**2))
print(f"RMS after Sum-of-Windows Norm: {rms_after_sow_norm:.4e}")

reconstructed_signal_final = reconstructed_signal_normalized.copy()

if rms_after_sow_norm > 1e-12:
    scale_factor = EXPECTED_RMS / rms_after_sow_norm
    reconstructed_signal_final *= scale_factor
    print(f"Applied final scaling factor {scale_factor:.4f} to match target RMS ({EXPECTED_RMS:.4e}).")
else:
    print("Warning: RMS after SoW normalization is near zero. Skipping final RMS scaling.")

rms_final = np.nan
max_abs_final = np.nan
if len(reconstructed_signal_final) > 0:
    rms_final = np.sqrt(np.mean(np.abs(reconstructed_signal_final)**2))
    max_abs_final = np.max(np.abs(reconstructed_signal_final))

print("\nSignal Stats AFTER Final Normalization:")
print(f"  Final RMS: {rms_final:.4e}")
print(f"  Final Max Abs: {max_abs_final:.4e}")

reconstructed_signal = reconstructed_signal_final.copy()
print("\nNormalization complete.")


# --- 7. Evaluation & Visualization ---
print("\n--- Evaluating Reconstruction ---")
print("Regenerating ground truth baseband for comparison...")
gt_duration = total_duration_recon
num_samples_gt_compare = int(round(gt_duration * GLBL_RECON_SAMPLE_RATE)) if GLBL_RECON_SAMPLE_RATE is not None else 0
t_gt_compare = np.linspace(0, gt_duration, num_samples_gt_compare, endpoint=False) if num_samples_gt_compare > 0 else np.array([])
gt_baseband_compare = np.zeros(num_samples_gt_compare, dtype=complex)
mod = global_attrs.get('modulation', 'qam16')
bw_gt = global_attrs.get('total_signal_bandwidth_hz', None)

if bw_gt is None or bw_gt <= 0:
     print(f"Error: Invalid ground truth bandwidth ({bw_gt}). Cannot regenerate GT.")
elif GLBL_RECON_SAMPLE_RATE is None or GLBL_RECON_SAMPLE_RATE <=0:
     print(f"Error: Invalid reconstruction sample rate ({GLBL_RECON_SAMPLE_RATE}). Cannot regenerate GT.")
elif num_samples_gt_compare <= 0:
     print(f"Error: Zero samples for GT comparison buffer.")
else:
    print(f"Attempting GT regeneration for modulation: {mod}, Bandwidth: {bw_gt/1e6:.2f} MHz")
    if mod.lower() == 'qam16':
        try:
            symbol_rate_gt = bw_gt / 4
            print(f"Using GT Symbol Rate = {symbol_rate_gt/1e6:.2f} Msps")
            num_symbols_gt = int(np.ceil(gt_duration * symbol_rate_gt))
            if num_symbols_gt > 0:
                symbols = (np.random.choice([-3,-1,1,3], size=num_symbols_gt) + 1j*np.random.choice([-3,-1,1,3], size=num_symbols_gt))/np.sqrt(10)
                samples_per_symbol_gt = max(1, int(round(GLBL_RECON_SAMPLE_RATE/symbol_rate_gt)))
                baseband_symbols = np.repeat(symbols, samples_per_symbol_gt)
                len_to_take = min(len(baseband_symbols), num_samples_gt_compare)
                gt_baseband_compare[:len_to_take] = baseband_symbols[:len_to_take]
                if len_to_take < num_samples_gt_compare:
                    print(f"Info: Generated GT ({len_to_take}) shorter than target ({num_samples_gt_compare}). Padded.")
            else: print("Warning: Calculated zero GT symbols.")
        except Exception as gt_gen_e:
             print(f"Error during QAM16 GT generation: {gt_gen_e}")
    else: print(f"Warning: GT regeneration not implemented for '{mod}'.")

    gt_rms_before = 0.0
    if len(gt_baseband_compare) > 0: gt_rms_before = np.sqrt(np.mean(np.abs(gt_baseband_compare)**2))
    print(f"GT RMS Before Scale: {gt_rms_before:.4e}")
    if gt_rms_before > 1e-20:
        gt_scale_factor = EXPECTED_RMS / gt_rms_before
        gt_baseband_compare *= gt_scale_factor
        gt_rms_after = np.sqrt(np.mean(np.abs(gt_baseband_compare)**2))
        print(f"Scaled GT baseband. Target RMS: {EXPECTED_RMS:.4e}, Actual RMS: {gt_rms_after:.4e}")
    else: print("Warning: GT baseband power near zero. Scaling skipped.")


# **** CORRECTED SPECTRUM FUNCTION AND PLOT LABEL ****
def compute_spectrum(signal, sample_rate):
    """
    Compute the power spectrum of a signal in dB.

    Args:
        signal (np.ndarray): Input signal (complex or real).
        sample_rate (float): Sampling frequency in Hz.

    Returns:
        freqs (np.ndarray): Frequency array (shifted for centered spectrum).
        spec_db (np.ndarray): Power spectrum in dB (shifted). Returns empty if input invalid.
    """
    signal = np.asarray(signal)
    n = len(signal)
    if n < 2 or sample_rate is None or sample_rate <= 0:
        print(f"Warning: Cannot compute spectrum with n={n}, sample_rate={sample_rate}")
        return np.array([]), np.array([])

    signal_complex = signal.astype(np.complex128)
    if not np.all(np.isfinite(signal_complex)):
         print("Warning: Non-finite values in signal for spectrum computation. Replacing with zeros.")
         signal_complex = np.nan_to_num(signal_complex)
    try:
        fft_result = np.fft.fft(signal_complex)
        # Compute Power Spectrum = |FFT|^2 / n
        spec = (np.abs(fft_result)**2) / n
        spec_db = 10 * np.log10(spec + 1e-20) # Epsilon to avoid log(0)
        freqs = np.fft.fftfreq(n, d=1/sample_rate)
        return np.fft.fftshift(freqs), np.fft.fftshift(spec_db)
    except Exception as spec_e:
        print(f"Error computing spectrum: {spec_e}")
        return np.array([]), np.array([])


# --- Evaluation Metrics Calculation ---
mse = np.inf
nmse = np.inf
evm = np.inf
reconstructed_signal_aligned = reconstructed_signal.copy() # Start with final signal

max_len_eval = min(len(reconstructed_signal), len(gt_baseband_compare))
valid_indices_for_eval = reliable_indices[(reliable_indices >= 0) & (reliable_indices < max_len_eval)] # Use reliable indices within bounds

print(f"\nCalculating metrics using {len(valid_indices_for_eval)} reliable samples (indices < {max_len_eval}).")
if len(valid_indices_for_eval) >= 10:
    gt_reliable = gt_baseband_compare[valid_indices_for_eval]
    recon_reliable = reconstructed_signal[valid_indices_for_eval]

    if np.all(np.isfinite(gt_reliable)) and np.all(np.isfinite(recon_reliable)):
        mean_power_gt_reliable = np.mean(np.abs(gt_reliable)**2)
        mean_power_recon_reliable = np.mean(np.abs(recon_reliable)**2)
        print(f"  Mean Power GT (Reliable Segments): {mean_power_gt_reliable:.4e}")
        print(f"  Mean Power Recon (Reliable Segments): {mean_power_recon_reliable:.4e}")

        if mean_power_gt_reliable > 1e-20 and mean_power_recon_reliable > 1e-20:
             power_scale_factor_for_plot = np.sqrt(mean_power_gt_reliable / mean_power_recon_reliable)
             reconstructed_signal_aligned = reconstructed_signal * power_scale_factor_for_plot # Scale final signal for plot alignment
             recon_reliable_scaled_for_evm = recon_reliable * power_scale_factor_for_plot
             print(f"Applied plotting alignment scale factor: {power_scale_factor_for_plot:.4f}")
             error_reliable = gt_reliable - recon_reliable_scaled_for_evm
             mse = np.mean(np.abs(error_reliable)**2)
             nmse = mse / mean_power_gt_reliable
             if nmse >= 0: evm = np.sqrt(nmse) * 100
             else: evm = np.inf; print("Warning: Negative NMSE.")
             rms_aligned_check = np.sqrt(np.mean(np.abs(reconstructed_signal_aligned)**2))
             print(f"  RMS of Aligned Recon (for plotting): {rms_aligned_check:.4e}")
        else: print("Warning: Near-zero power in reliable segments. Cannot calculate metrics/align.")
    else: print("Warning: Non-finite values in reliable segments. Cannot calculate metrics.")
else: print(f"Warning: Not enough reliable samples ({len(valid_indices_for_eval)}) for evaluation.")


print("\nEvaluation Metrics:")
print(f"  MSE : {mse:.4e}")
if np.isfinite(nmse) and nmse > 1e-20: print(f"  NMSE: {nmse:.4e} ({10*np.log10(nmse):.2f} dB)")
elif np.isfinite(nmse): print(f"  NMSE: {nmse:.4e} (dB invalid)")
else: print(f"  NMSE: {nmse}")
print(f"  EVM : {evm:.2f}%" if np.isfinite(evm) else f"  EVM : {evm}")


# --- Plotting ---
print("\n--- Generating Matplotlib Plots ---")
plt.style.use('seaborn-v0_8-darkgrid')
fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
plot_samples = min(plot_length, len(t_gt_compare), len(gt_baseband_compare), len(reconstructed_signal_aligned))

if plot_samples > 0:
    time_axis_plot = t_gt_compare[:plot_samples] * 1e6 # Time in microseconds
    # Plot GT
    gt_plot_data = gt_baseband_compare[:plot_samples]
    axs[0].plot(time_axis_plot, np.real(gt_plot_data), label='GT (Real)', linewidth=1.0)
    axs[0].plot(time_axis_plot, np.imag(gt_plot_data), label='GT (Imag)', alpha=0.7, linewidth=1.0)
    axs[0].set_title(f'Ground Truth (Target RMS={EXPECTED_RMS:.2e})')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend(fontsize='small', loc='upper right'); axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ylim_gt_abs = EXPECTED_RMS * 4
    if np.any(gt_plot_data): max_abs_gt_plot = np.max(np.abs(gt_plot_data)); ylim_gt_abs = max(ylim_gt_abs, max_abs_gt_plot)
    axs[0].set_ylim(-ylim_gt_abs * 1.2, ylim_gt_abs * 1.2)
    # Plot Recon (Aligned)
    recon_plot_data = reconstructed_signal_aligned[:plot_samples]
    recon_plot_data_real = np.nan_to_num(np.real(recon_plot_data))
    recon_plot_data_imag = np.nan_to_num(np.imag(recon_plot_data))
    axs[1].plot(time_axis_plot, recon_plot_data_real, label='Recon (Real)', linewidth=1.0)
    axs[1].plot(time_axis_plot, recon_plot_data_imag, label='Recon (Imag)', alpha=0.7, linewidth=1.0)
    title_recon = f'Reconstructed Signal (Plot Aligned)'
    if np.isfinite(evm): title_recon += f' / Eval EVM: {evm:.2f}%'
    axs[1].set_title(title_recon); axs[1].set_ylabel('Amplitude')
    axs[1].legend(fontsize='small', loc='upper right'); axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    ylim_recon_abs = ylim_gt_abs
    if np.any(recon_plot_data): max_abs_recon_plot = np.max(np.abs(recon_plot_data)); ylim_recon_abs = max(ylim_gt_abs, max_abs_recon_plot)
    axs[1].set_ylim(-ylim_recon_abs * 1.2, ylim_recon_abs * 1.2)
    # Plot Error
    error_signal = gt_plot_data - recon_plot_data
    error_signal_real = np.nan_to_num(np.real(error_signal))
    error_signal_imag = np.nan_to_num(np.imag(error_signal))
    axs[2].plot(time_axis_plot, error_signal_real, label='Error (Real)', linewidth=1.0)
    axs[2].plot(time_axis_plot, error_signal_imag, label='Error (Imag)', alpha=0.7, linewidth=1.0)
    axs[2].set_title('Error Signal (GT - Plot Aligned Recon)')
    axs[2].set_xlabel('Time (µs)'); axs[2].set_ylabel('Amplitude')
    axs[2].legend(fontsize='small', loc='upper right'); axs[2].grid(True, which='both', linestyle='--', linewidth=0.5)
    ylim_err_abs = ylim_recon_abs * 0.5
    if np.any(error_signal): max_abs_err_plot = np.max(np.abs(error_signal)); ylim_err_abs = max(ylim_err_abs, max_abs_err_plot)
    axs[2].set_ylim(-ylim_err_abs * 1.2, ylim_err_abs * 1.2)
    plt.tight_layout(); plt.show(block=False); plt.pause(0.1)
else: print("Skipping time domain plots: Not enough samples ({plot_samples} samples).")

# Spectrum Plot
plt.figure(figsize=(12, 7))
plot_spectrum_flag = False
# Plot GT spectrum
if len(gt_baseband_compare) > 1 and np.all(np.isfinite(gt_baseband_compare)) and GLBL_RECON_SAMPLE_RATE is not None:
    f_gt, spec_gt_db = compute_spectrum(gt_baseband_compare, GLBL_RECON_SAMPLE_RATE)
    if len(f_gt) > 0: plt.plot(f_gt / 1e6, spec_gt_db, label='GT Spectrum', alpha=0.8, linewidth=1.5); plot_spectrum_flag = True
    else: print("GT spectrum computation returned empty.")
else: print("Skipping GT spectrum plot.")
# Plot Reconstructed spectrum (using aligned signal)
if len(reconstructed_signal_aligned) > 1 and np.all(np.isfinite(reconstructed_signal_aligned)) and GLBL_RECON_SAMPLE_RATE is not None:
    f_recon, spec_recon_db = compute_spectrum(reconstructed_signal_aligned, GLBL_RECON_SAMPLE_RATE)
    if len(f_recon) > 0: plt.plot(f_recon / 1e6, spec_recon_db, label='Recon Spectrum (Plot Aligned)', ls='--', alpha=0.9, linewidth=1.5); plot_spectrum_flag = True
    else: print("Reconstructed spectrum computation returned empty.")
else: print("Skipping Reconstructed spectrum plot.")

if plot_spectrum_flag:
    plt.title('Power Spectrum Comparison') # **** CHANGED TITLE ****
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power Spectrum (dB)') # **** CHANGED LABEL ****
    plt.ylim(bottom=spectrum_ylim)
    max_spec_val = -np.inf
    ax = plt.gca();
    for line in ax.get_lines(): ydata = line.get_ydata(); max_spec_val = max(max_spec_val, np.max(ydata[np.isfinite(ydata)])) if len(ydata)>0 and np.any(np.isfinite(ydata)) else max_spec_val
    if np.isfinite(max_spec_val): plt.ylim(bottom=spectrum_ylim, top=min(max_spec_val + 10, 20))
    else: plt.ylim(bottom=spectrum_ylim, top=0) # Adjust default top if needed
    plt.legend(); plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.tight_layout(); plt.show() # Show final plot blocking
else: print("Skipping spectrum plot.")

print("\nScript finished.")