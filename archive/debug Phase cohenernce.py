import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import h5py
from tqdm import tqdm
import sys
import plotly.graph_objects as go # Import Plotly

# --- Parameters ---
input_filename = "simulated_chunks_25GHz_400MHzBW_qam16_sdr56MHz.h5"
stitching_window_type = 'hann' # Options: 'hann', 'hamming', 'blackman', 'boxcar' etc.
plot_length = 5000  # Number of samples to plot in final time-domain plots
spectrum_ylim = -100 # Lower Y limit for spectrum plots in dB
EXPECTED_RMS = 1.39e-02  # Target RMS value after initial scaling

# --- Helper function for spectrum calculation (to avoid repetition) ---
def compute_spectrum(signal_data, sample_rate):
    """Calculates the FFT spectrum, shifts it, converts to dB, and normalizes."""
    n = len(signal_data)
    if n < 2: # Need at least 2 points for FFT
        return np.array([]), np.array([])
    f = np.fft.fftshift(np.fft.fftfreq(n, d=1/sample_rate))
    s = np.fft.fftshift(np.fft.fft(signal_data))
    # Calculate magnitude spectrum in dB, adding epsilon for log(0)
    spec_mag = np.abs(s)
    spec_db = 20 * np.log10(spec_mag + 1e-12) # Add epsilon to avoid log10(0)
    # Normalize to peak value (0 dB) - use nanmax to handle potential NaNs safely
    max_db = np.nanmax(spec_db)
    if np.isfinite(max_db):
        spec_db -= max_db
    else:
        print("Warning: Max spectrum value is non-finite. Not normalizing spectrum.")
    return f, spec_db

# --- 1. Load Simulated Chunk Data ---
print(f"Loading data from: {input_filename}")
loaded_chunks = []
loaded_metadata = []
global_attrs = {}
try:
    with h5py.File(input_filename, 'r') as f:
        # Load global attributes
        for key, value in f.attrs.items(): global_attrs[key] = value
        print("--- Global Parameters ---")
        for key, value in global_attrs.items(): print(f"{key}: {value}")
        print("-------------------------")

        actual_chunks = global_attrs.get('actual_num_chunks_saved', 0)
        if actual_chunks == 0: raise ValueError("HDF5 attribute 'actual_num_chunks_saved' indicates 0 chunks.")

        # Load chunk data and metadata
        for i in range(actual_chunks):
            group_name = f'chunk_{i:03d}'
            if group_name not in f:
                print(f"Warning: Group '{group_name}' not found in HDF5 file. Skipping.")
                continue
            group = f[group_name]
            if 'iq_data' not in group:
                print(f"Warning: 'iq_data' dataset not found in group '{group_name}'. Skipping.")
                continue

            chunk_data = group['iq_data'][:].astype(np.complex128)
            meta = {key: value for key, value in group.attrs.items()}
            loaded_chunks.append(chunk_data)
            loaded_metadata.append(meta)

except Exception as e:
    print(f"Error loading HDF5 file '{input_filename}': {e}"); sys.exit(1)

if not loaded_chunks: print("No chunk data loaded. Exiting."); sys.exit(1)
print(f"\nLoaded {len(loaded_chunks)} chunks.")

# Extract essential parameters from global attributes
fs_sdr = global_attrs.get('sdr_sample_rate_hz', None)
fs_recon = global_attrs.get('ground_truth_sample_rate_hz', None)
overlap_factor = global_attrs.get('overlap_factor', 0.1) # Default overlap 10%
tuning_delay = global_attrs.get('tuning_delay_s', 5e-6) # Default tuning delay 5us

if fs_sdr is None or fs_recon is None:
    print("Error: Sample rate information ('sdr_sample_rate_hz' or 'ground_truth_sample_rate_hz') missing in HDF5 attributes."); sys.exit(1)
if not isinstance(fs_sdr, (int, float)) or fs_sdr <= 0:
     print(f"Error: Invalid SDR sample rate retrieved: {fs_sdr}"); sys.exit(1)
if not isinstance(fs_recon, (int, float)) or fs_recon <= 0:
     print(f"Error: Invalid Reconstruction sample rate retrieved: {fs_recon}"); sys.exit(1)


# --- 1b. Correct Initial Amplitude Scaling of Loaded Chunks ---
print("\n--- Correcting Initial Amplitude Scaling of Loaded Chunks ---")
scaled_loaded_chunks = []
print("Chunk | RMS Before Scaling | Scaling Factor | RMS After Scaling | Max Abs (After)")
print("------|------------------|----------------|-------------------|----------------")
overall_scaling_successful = True # Track if any chunk fails
for i, chunk in enumerate(loaded_chunks):
    chunk_len = len(chunk)
    if chunk_len == 0:
        scaled_loaded_chunks.append(chunk) # Append empty chunk
        print(f"{i:<5d} | --- EMPTY ---    | ---            | --- EMPTY ---     | ---")
        continue

    # Check for non-finite values *before* calculations
    if not np.all(np.isfinite(chunk)):
        print(f"ERROR: Chunk {i} contains non-finite values BEFORE scaling.")
        overall_scaling_successful = False
        scaled_loaded_chunks.append(chunk) # Append original problematic chunk
        continue

    # Calculate RMS safely
    # Using np.mean(np.real(chunk * np.conj(chunk))) is equivalent to np.mean(np.abs(chunk)**2)
    mean_sq = np.mean(np.real(chunk * np.conj(chunk)))
    rms_before_scaling = np.sqrt(mean_sq) if mean_sq >= 0 else 0 # Handle potential negative mean_sq from floating point issues

    # Check if RMS is too small (effectively zero)
    if rms_before_scaling < 1e-12:
        print(f"{i:<5d} | {rms_before_scaling:.4e}       | SKIPPED (Zero) | {rms_before_scaling:.4e}        | {np.max(np.abs(chunk)):.4e}")
        scaled_loaded_chunks.append(chunk) # Append the original (near) zero chunk
        continue

    # Calculate scaling factor
    scaling_factor = EXPECTED_RMS / rms_before_scaling

    # Apply scaling factor
    scaled_chunk = (chunk * scaling_factor).astype(np.complex128)

    # Check for non-finite values *after* scaling
    if not np.all(np.isfinite(scaled_chunk)):
         print(f"ERROR: Chunk {i} contains non-finite values AFTER scaling (Factor: {scaling_factor:.4f}). RMS before: {rms_before_scaling:.4e}")
         overall_scaling_successful = False
         scaled_loaded_chunks.append(scaled_chunk) # Append chunk with NaNs for inspection
         continue

    # Verify RMS after scaling
    rms_after_scaling = np.sqrt(np.mean(np.real(scaled_chunk * np.conj(scaled_chunk))))
    max_abs_after = np.max(np.abs(scaled_chunk))

    # Use a relative tolerance for checking RMS
    if not np.isclose(rms_after_scaling, EXPECTED_RMS, rtol=1e-3):
        print(f"WARNING: Chunk {i} RMS scaling mismatch ({rms_after_scaling:.4e} vs {EXPECTED_RMS:.4e})")

    print(f"{i:<5d} | {rms_before_scaling:.4e}       | {scaling_factor:<14.4f} | {rms_after_scaling:.4e}        | {max_abs_after:.4e}")
    scaled_loaded_chunks.append(scaled_chunk)

if not overall_scaling_successful:
    print("\nERROR: Non-finite values detected or generated during initial scaling. Proceeding, but results may be compromised.")
    # Decide whether to exit: sys.exit(1) # Uncomment to stop execution on scaling errors
print("--- Initial Amplitude Scaling Complete ---")


# --- SKIPPING AF ---
print("\n--- Skipping Adaptive Filtering ---")
# If AF was implemented, it would go here.
aligned_chunks = scaled_loaded_chunks # Use the scaled chunks directly for the next step


# --- SKIPPING CP ---
print("\n--- Skipping CP Decomposition ---")
# If CP Decomposition was implemented, it would go here.


# --- 2. Upsample Chunks (Multi-Stage) ---
print("\n--- Upsampling chunks using MULTI-STAGE resample_poly ---")
# Determine intermediate sample rate - aim for something between fs_sdr and fs_recon
# but also consider integer factors if possible. Max 5x SDR rate is often reasonable.
fs_interim = min(fs_sdr * 5, fs_recon)
# Simple heuristic: if fs_recon is much larger than fs_sdr*5, maybe pick fs_recon/2 ?
# Or just use fs_sdr * 5 as a reasonable upper bound for the first stage.
print(f"Intermediate sample rate for resampling: {fs_interim / 1e6:.2f} MHz")

upsampled_chunks = []
debug_rms_upsample_input = []
debug_rms_upsample_output = []
debug_max_abs_upsample_input = []
debug_max_abs_upsample_output = []
upsampling_successful = True # Track success within this block

for i, chunk_data in tqdm(enumerate(aligned_chunks), total=len(aligned_chunks), desc="Upsampling"):
    # Get corresponding metadata if needed (not currently used in resampling)
    # meta = loaded_metadata[i]

    # Calculate expected output length based on original duration and target rate
    chunk_duration_sdr = len(chunk_data)/fs_sdr if fs_sdr > 0 else 0
    num_samples_chunk_recon = int(round(chunk_duration_sdr * fs_recon))

    # Handle empty or very short chunks which resample_poly cannot process
    if len(chunk_data) < 2:
        print(f"Warning: Chunk {i} is too short ({len(chunk_data)} samples) for resampling. Appending zeros ({num_samples_chunk_recon} samples).")
        upsampled_chunks.append(np.zeros(num_samples_chunk_recon, dtype=np.complex128))
        debug_rms_upsample_input.append(0.0)
        debug_rms_upsample_output.append(0.0)
        debug_max_abs_upsample_input.append(0.0)
        debug_max_abs_upsample_output.append(0.0)
        continue

    try:
        # --- Calculate RMS/Max before upsampling ---
        rms_in = np.nan
        max_abs_in = np.nan
        if np.all(np.isfinite(chunk_data)):
             rms_in = np.sqrt(np.mean(np.real(chunk_data * np.conj(chunk_data))))
             max_abs_in = np.max(np.abs(chunk_data))
        else:
             print(f"Warning: Chunk {i} has non-finite values BEFORE upsampling.")
             upsampling_successful = False # Mark potential issue

        debug_rms_upsample_input.append(rms_in)
        debug_max_abs_upsample_input.append(max_abs_in)

        # Skip resampling if input is non-finite or effectively zero
        if not np.isfinite(rms_in) or rms_in < 1e-12:
            print(f"Skipping resampling for chunk {i} due to zero/NaN input RMS.")
            upsampled_chunks.append(np.zeros(num_samples_chunk_recon, dtype=np.complex128))
            debug_rms_upsample_output.append(0.0)
            debug_max_abs_upsample_output.append(0.0)
            continue

        # --- Stage 1: Upsample to Intermediate Rate ---
        # Calculate integer up/down factors, simplifying by GCD
        up1 = int(round(fs_interim))
        down1 = int(round(fs_sdr))
        common1 = np.gcd(up1, down1)
        up1 //= common1
        down1 //= common1
        # print(f"Stage 1 (Chunk {i}): up={up1}, down={down1}") # Debug
        interim_chunk = sig.resample_poly(chunk_data, up1, down1)

        # --- Stage 2: Upsample from Intermediate Rate to Final Rate ---
        up2 = int(round(fs_recon))
        down2 = int(round(fs_interim))
        common2 = np.gcd(up2, down2)
        up2 //= common2
        down2 //= common2
        # print(f"Stage 2 (Chunk {i}): up={up2}, down={down2}") # Debug
        upsampled_chunk = sig.resample_poly(interim_chunk, up2, down2)

        # --- Adjust length to match expected output length ---
        current_len = len(upsampled_chunk)
        if current_len != num_samples_chunk_recon:
             # print(f"Debug: Chunk {i} length mismatch after resample. Expected {num_samples_chunk_recon}, got {current_len}. Adjusting.") # Debug
             if current_len > num_samples_chunk_recon:
                 upsampled_chunk = upsampled_chunk[:num_samples_chunk_recon]
             else: # current_len < num_samples_chunk_recon
                 # Pad with zeros - consider edge effects, maybe pad with last value? Zeros are safer.
                 upsampled_chunk = np.pad(upsampled_chunk, (0, num_samples_chunk_recon - current_len), mode='constant')

        # Ensure correct data type
        upsampled_chunk = upsampled_chunk.astype(np.complex128)

        # --- Calculate RMS/Max after upsampling ---
        rms_out = np.nan
        max_abs_out = np.nan
        if np.all(np.isfinite(upsampled_chunk)):
            rms_out = np.sqrt(np.mean(np.real(upsampled_chunk * np.conj(upsampled_chunk))))
            max_abs_out = np.max(np.abs(upsampled_chunk))
        else:
            print(f"ERROR: Chunk {i} has non-finite values AFTER upsampling stage 2.")
            upsampling_successful = False
            # What to do? Append zeros? Append the chunk with NaNs? Let's append zeros.
            upsampled_chunk = np.zeros(num_samples_chunk_recon, dtype=np.complex128)
            rms_out = 0.0
            max_abs_out = 0.0

        debug_rms_upsample_output.append(rms_out)
        debug_max_abs_upsample_output.append(max_abs_out)
        upsampled_chunks.append(upsampled_chunk.copy()) # Append the result (or zeros if error)

        # --- Plotting first chunk (optional, keep if useful for debugging) ---
        if i == 0:
            print("Plotting first upsampled chunk (MULTI-STAGE)...")
            plt.figure(figsize=(12,4))
            time_axis_debug = np.arange(len(upsampled_chunk))/fs_recon*1e6 # Time in us
            plt.plot(time_axis_debug, np.real(upsampled_chunk), label='Real')
            plt.plot(time_axis_debug, np.imag(upsampled_chunk), label='Imag', alpha=0.7)
            plt.title(f'First Upsampled Chunk (RMS={rms_out:.3e}, MaxAbs={max_abs_out:.3e})')
            plt.xlabel('Time (µs)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True)
            # Sensible x-axis limit (e.g., first 5 microseconds or full chunk if shorter)
            plot_duration_us = 5.0
            max_time_us = len(upsampled_chunk) / fs_recon * 1e6
            xlim_upper = min(plot_duration_us, max_time_us)
            plt.xlim(-0.1 * xlim_upper, xlim_upper) # Show a bit before t=0
            # Dynamic Y-axis limit based on max abs value
            ylim_abs = max(max_abs_out * 1.2, 0.05) # Ensure some minimum range if signal is tiny
            plt.ylim(-ylim_abs, ylim_abs)
            plt.show()

            print("Plotting Multi-Stage Spectra Comparison...")
            plt.figure(figsize=(12, 6))
            # Before Upsampling
            f_before, spec_before_db = compute_spectrum(chunk_data, fs_sdr)
            if len(f_before) > 0:
                 plt.plot(f_before/1e6, spec_before_db, label=f'BEFORE (Fs={fs_sdr/1e6:.1f}MHz)', alpha=0.8)
            # After Upsampling
            f_after, spec_after_db = compute_spectrum(upsampled_chunk, fs_recon)
            if len(f_after) > 0:
                 plt.plot(f_after/1e6, spec_after_db, label=f'AFTER (Fs={fs_recon/1e6:.1f}MHz)', ls='--', alpha=0.8)

            plt.title('Spectra Comparison Upsampling (Peak Normalized)')
            plt.xlabel('Frequency (MHz relative to Chunk Center)')
            plt.ylabel('Magnitude (dB)')
            plt.ylim(-120, 5) # Show spectrum down to -120 dB, peak at 0 dB
            plt.legend()
            plt.grid(True)
            plt.show()

    except Exception as resample_e:
        print(f"Error during resampling chunk {i}: {resample_e}. Appending zeros.")
        upsampled_chunks.append(np.zeros(num_samples_chunk_recon, dtype=np.complex128))
        # Record failure in debug lists
        debug_rms_upsample_input.append(rms_in) # May be NaN if it failed early
        debug_max_abs_upsample_input.append(max_abs_in) # May be NaN
        debug_rms_upsample_output.append(np.nan)
        debug_max_abs_upsample_output.append(np.nan)
        upsampling_successful = False # Mark failure


# --- Post-Upsampling Checks ---
if not upsampling_successful:
     print("\nWARNING: Errors or non-finite values occurred during upsampling. Subsequent results may be unreliable.")

# Print summary table
print("\n--- RMS and Max Abs Before/After Upsampling ---")
print("Chunk | RMS Before | Max Abs Before | RMS After  | Max Abs After")
print("------|------------|----------------|------------|--------------")
min_len_debug = min(len(debug_rms_upsample_input), len(debug_max_abs_upsample_input),
                    len(debug_rms_upsample_output), len(debug_max_abs_upsample_output),
                    len(aligned_chunks)) # Ensure index 'i' is valid for aligned_chunks
for i in range(min_len_debug):
     rms_in_str = f"{debug_rms_upsample_input[i]:.4e}" if np.isfinite(debug_rms_upsample_input[i]) else "  NaN     "
     max_in_str = f"{debug_max_abs_upsample_input[i]:.4e}" if np.isfinite(debug_max_abs_upsample_input[i]) else "  NaN     "
     rms_out_str = f"{debug_rms_upsample_output[i]:.4e}" if np.isfinite(debug_rms_upsample_output[i]) else "  NaN     "
     max_out_str = f"{debug_max_abs_upsample_output[i]:.4e}" if np.isfinite(debug_max_abs_upsample_output[i]) else "  NaN     "
     print(f"{i:<5d} | {rms_in_str} | {max_in_str}   | {rms_out_str} | {max_out_str}")


# --- 3. Stitching (Overlap-Add) with Phase Alignment ---
print("\n--- Stitching Upsampled Chunks (Overlap-Add) with Phase Alignment ---")
if not upsampled_chunks:
    print("Error: No upsampled chunks available for stitching.")
    sys.exit(1)

# +++ FIX: Calculate the original chunk duration at SDR rate +++
# This is needed to determine the time advance between chunks.
# Assume all original chunks had the same length as the first one.
if len(loaded_chunks) > 0 and len(loaded_chunks[0]) > 0 and fs_sdr > 0:
    chunk_duration_sdr_s = len(loaded_chunks[0]) / fs_sdr
    print(f"Calculated original chunk duration (SDR): {chunk_duration_sdr_s * 1e6:.2f} us")
else:
    print("Error: Cannot determine original chunk duration from loaded_chunks or fs_sdr. Exiting.")
    sys.exit(1) # Need this duration to proceed
# +++ END FIX +++

# Calculate expected total duration and samples more carefully
# Duration of the first upsampled chunk (at the reconstruction rate)
chunk_duration_recon_s = len(upsampled_chunks[0]) / fs_recon if fs_recon > 0 and len(upsampled_chunks[0]) > 0 else 0
# Time between the start of consecutive chunks (based on SDR duration, overlap, and delay)
time_advance_per_chunk = chunk_duration_sdr_s * (1.0 - overlap_factor) + tuning_delay # <--- Use chunk_duration_sdr_s
# Total duration = duration of first chunk + (N-1) advances
total_duration_recon = chunk_duration_recon_s + (len(loaded_chunks) - 1) * time_advance_per_chunk
num_samples_recon = int(round(total_duration_recon * fs_recon))

# Sanity check on calculated reconstruction length
if num_samples_recon <= 0:
    print(f"Error: Calculated reconstruction samples is non-positive ({num_samples_recon}). Check parameters.")
    print(f"  Chunk duration (SDR): {chunk_duration_sdr_s} s")
    print(f"  Chunk duration (Recon): {chunk_duration_recon_s} s")
    print(f"  Time advance per chunk: {time_advance_per_chunk} s")
    print(f"  Num loaded chunks: {len(loaded_chunks)}")
    sys.exit(1)

# Initialize arrays for reconstruction
reconstructed_signal = np.zeros(num_samples_recon, dtype=np.complex128)
sum_of_windows = np.zeros(num_samples_recon, dtype=np.float64) # Use float64 for better precision

print(f"Reconstruction target: {num_samples_recon} samples @ {fs_recon/1e6:.2f} MHz (Duration: {total_duration_recon*1e6:.1f} us)")

# Initialize time tracking for stitching
current_recon_time_start = 0.0 # Start time of the current chunk in the reconstruction timeline (seconds)

# --- Phase Alignment Stage ---
print("Performing Phase Alignment...")
# Calculate overlap samples at the reconstruction rate
overlap_samples = int(round(chunk_duration_sdr_s * overlap_factor * fs_recon)) # <--- Use chunk_duration_sdr_s
if overlap_samples < 10: # Warn if overlap is too small for reliable correlation
    print(f"Warning: Overlap region is small ({overlap_samples} samples). Phase alignment might be unreliable.")
overlap_samples = max(1, overlap_samples) # Ensure it's at least 1

phase_corrected_chunks = []
if len(upsampled_chunks) > 0:
     # Handle the first chunk (no previous chunk to align to)
     first_chunk = upsampled_chunks[0]
     if np.all(np.isfinite(first_chunk)):
        phase_corrected_chunks.append(first_chunk.copy())
     else:
         print("Warning: First upsampled chunk contains non-finite values. Replacing with zeros for stitching.")
         phase_corrected_chunks.append(np.zeros_like(first_chunk))

# Iterate through remaining chunks to align each to the *previously corrected* one
for i in range(1, len(upsampled_chunks)):
    # Get the previously corrected chunk and the current original upsampled chunk
    if not phase_corrected_chunks: # Should not happen if first chunk was handled
         print(f"Error: phase_corrected_chunks list is empty at iteration {i}. Stopping phase alignment.")
         break
    prev_chunk_corrected = phase_corrected_chunks[-1]
    curr_chunk_original = upsampled_chunks[i]

    # Basic checks before attempting alignment
    if len(prev_chunk_corrected) < overlap_samples or len(curr_chunk_original) < overlap_samples:
        print(f"Warning: Skipping phase alignment for chunk {i} due to insufficient length for overlap (Prev:{len(prev_chunk_corrected)}, Curr:{len(curr_chunk_original)}, Need:{overlap_samples}).")
        # Append the current chunk without correction (or zeros if it's bad)
        if np.all(np.isfinite(curr_chunk_original)):
             phase_corrected_chunks.append(curr_chunk_original.copy())
        else:
             print(f"Warning: Chunk {i} (original) contains non-finite values. Appending zeros.")
             phase_corrected_chunks.append(np.zeros_like(curr_chunk_original))
        continue

    # Extract overlap regions
    prev_overlap = prev_chunk_corrected[-overlap_samples:]
    curr_overlap = curr_chunk_original[:overlap_samples]

    # Check for non-finite values or zero signals in overlap regions
    if not np.all(np.isfinite(prev_overlap)) or not np.all(np.isfinite(curr_overlap)):
         print(f"Warning: Skipping phase alignment for chunk {i} due to non-finite values in overlap region.")
         if np.all(np.isfinite(curr_chunk_original)): phase_corrected_chunks.append(curr_chunk_original.copy())
         else: phase_corrected_chunks.append(np.zeros_like(curr_chunk_original))
         continue
    if np.allclose(prev_overlap, 0) or np.allclose(curr_overlap, 0):
         print(f"Warning: Skipping phase alignment for chunk {i} due to zero signal in overlap region.")
         if np.all(np.isfinite(curr_chunk_original)): phase_corrected_chunks.append(curr_chunk_original.copy())
         else: phase_corrected_chunks.append(np.zeros_like(curr_chunk_original))
         continue

    # --- Calculate Phase Offset ---
    try:
        # Calculate complex correlation at zero lag using mean (more robust to noise)
        # We want angle( Sum[ prev[n] * conj(curr[n]) ] )
        complex_correlation_zero_lag = np.mean(prev_overlap * np.conj(curr_overlap))
        phase_offset = np.angle(complex_correlation_zero_lag)

        # Apply correction: Multiply current chunk by exp(-1j * phase_offset)
        corrected_chunk = curr_chunk_original * np.exp(-1j * phase_offset)

        # Check if correction introduced non-finite values
        if not np.all(np.isfinite(corrected_chunk)):
            print(f"Warning: Non-finite values after phase correction for chunk {i}. Reverting to original.")
            if np.all(np.isfinite(curr_chunk_original)): phase_corrected_chunks.append(curr_chunk_original.copy())
            else: phase_corrected_chunks.append(np.zeros_like(curr_chunk_original))
        else:
            # print(f"Chunk {i}: Applied phase correction: {np.degrees(phase_offset):.2f} deg") # Debug
            phase_corrected_chunks.append(corrected_chunk)

    except Exception as phase_err:
         print(f"Error during phase alignment calculation for chunk {i}: {phase_err}. Appending original chunk.")
         if np.all(np.isfinite(curr_chunk_original)): phase_corrected_chunks.append(curr_chunk_original.copy())
         else: phase_corrected_chunks.append(np.zeros_like(curr_chunk_original))

# --- Overlap-Add Stage using Phase Corrected Chunks ---
print("Performing Overlap-Add Stitching...")
stitching_successful = True # Track success within this block

for i, chunk_to_add in tqdm(enumerate(phase_corrected_chunks), total=len(phase_corrected_chunks), desc="Overlap-Add Stitching"):
    # Basic check for valid chunk data
    if chunk_to_add is None or len(chunk_to_add) == 0:
        print(f"Skipping empty or invalid phase-corrected chunk {i}.")
        # Advance time if not the last chunk
        if i < len(phase_corrected_chunks) - 1:
             current_recon_time_start += time_advance_per_chunk
        continue

    # Check for non-finite values before adding
    if not np.all(np.isfinite(chunk_to_add)):
        print(f"*** WARNING: Non-finite phase-corrected chunk {i} DETECTED BEFORE adding. Replacing with zeros. ***")
        chunk_to_add = np.zeros_like(chunk_to_add) # Replace with zeros
        stitching_successful = False # Mark potential issue

    # Calculate start/end indices in the reconstruction buffer
    start_idx_recon = int(round(current_recon_time_start * fs_recon))
    num_samples_in_chunk = len(chunk_to_add)
    end_idx_recon = start_idx_recon + num_samples_in_chunk

    # --- Boundary Checks ---
    # Check if start index is valid
    if start_idx_recon < 0:
        print(f"Warning: Chunk {i} calculated start index {start_idx_recon} is negative. Skipping chunk.")
        stitching_successful = False
        if i < len(phase_corrected_chunks) - 1: current_recon_time_start += time_advance_per_chunk
        continue

    # Check if chunk extends beyond the allocated buffer and truncate if necessary
    actual_len_in_buffer = num_samples_in_chunk
    if end_idx_recon > num_samples_recon:
        print(f"Warning: Chunk {i} extends beyond calculated total samples ({end_idx_recon} > {num_samples_recon}). Truncating chunk.")
        end_idx_recon = num_samples_recon
        actual_len_in_buffer = end_idx_recon - start_idx_recon # Recalculate length to use from chunk
        if actual_len_in_buffer < 0: actual_len_in_buffer = 0 # Ensure non-negative
    # Check if the segment to write has non-positive length after potential truncation
    if actual_len_in_buffer <= 0:
        # print(f"Skipping chunk {i} due to zero or negative length segment ({actual_len_in_buffer}) after boundary checks at start index {start_idx_recon}.") # Debug
        # Advance time for the next chunk if this wasn't the last one
        if i < len(phase_corrected_chunks) - 1:
            current_recon_time_start += time_advance_per_chunk
        continue

    # --- Calculate RMS and Max Abs *before* windowing (using the potentially truncated length) ---
    segment_data = chunk_to_add[:actual_len_in_buffer]
    # Re-check segment_data for safety, though chunk_to_add should be clean now
    if np.all(np.isfinite(segment_data)):
        rms_in_stitching = np.sqrt(np.mean(np.real(segment_data * np.conj(segment_data))))
        max_abs_chunk = np.max(np.abs(segment_data)) if len(segment_data) > 0 else 0
        # print(f"  Stitching Chunk {i}: Start {start_idx_recon}, End {end_idx_recon}, Len {actual_len_in_buffer}, RMS = {rms_in_stitching:.4e}, MaxAbs = {max_abs_chunk:.4e}") # Verbose Debug
    else:
        print(f"*** ERROR: Non-finite values detected in chunk {i} segment just before windowing! Skipping add. ***")
        rms_in_stitching = np.nan
        max_abs_chunk = np.nan
        stitching_successful = False
        if i < len(phase_corrected_chunks) - 1: current_recon_time_start += time_advance_per_chunk
        continue # Skip adding this corrupted chunk

    # --- Get and Normalize Window ---
    if actual_len_in_buffer < 2: # get_window needs at least 2 samples
        window = np.ones(actual_len_in_buffer, dtype=np.float64)
    else:
        window = sig.get_window(stitching_window_type, actual_len_in_buffer).astype(np.float64)
        # Normalize window to have RMS of 1 (helps preserve power, though not perfectly for all windows)
        window_rms = np.sqrt(np.mean(window**2))
        if window_rms > 1e-12: # Avoid division by zero
             window /= window_rms
        else:
            # print(f"Warning: Window RMS is near zero for chunk {i}. Using uniform window.") # Debug
            window = np.ones(actual_len_in_buffer, dtype=np.float64) # Fallback

    # --- Apply Window and Add to Signal ---
    try:
        segment_to_add = segment_data * window

        # Check for non-finite values *after* windowing
        if not np.all(np.isfinite(segment_to_add)):
            print(f"*** WARNING: Non-finite segment_to_add detected for chunk {i} AFTER windowing. Zeroing NaNs/Infs. ***")
            segment_to_add[~np.isfinite(segment_to_add)] = 0 # Replace bad values with 0
            stitching_successful = False

        # Add to reconstructed signal and sum of windows buffers
        reconstructed_signal[start_idx_recon:end_idx_recon] += segment_to_add
        sum_of_windows[start_idx_recon:end_idx_recon] += window

        # Check if the *result* in reconstructed_signal became non-finite (e.g., overflow)
        if not np.all(np.isfinite(reconstructed_signal[start_idx_recon:end_idx_recon])):
            print(f"*** CRITICAL WARNING: Non-finite values in reconstructed_signal after adding chunk {i}! Attempting to zero segment. ***")
            reconstructed_signal[start_idx_recon:end_idx_recon] = 0 # Try to recover by zeroing out problem area
            # This segment in sum_of_windows might also be problematic now, but leave it for normalization check
            stitching_successful = False

    except IndexError as idx_e:
         print(f"Error (IndexError) during overlap-add chunk {i}: {idx_e}")
         print(f"  Indices: start={start_idx_recon}, end={end_idx_recon}, recon_len={num_samples_recon}, actual_len={actual_len_in_buffer}")
         stitching_successful = False
         # Fall through to advance time
    except Exception as add_e:
        print(f"Error (General) during overlap-add chunk {i}: {add_e}")
        stitching_successful = False
        # Fall through to advance time

    # Advance the start time for the next chunk
    if i < len(phase_corrected_chunks) - 1:
        current_recon_time_start += time_advance_per_chunk

# --- End of Stitching Loop ---
print("\nOverlap-Add Stitching loop complete.")
if not stitching_successful:
    print("WARNING: Errors or non-finite values encountered during stitching. Normalization and evaluation may be unreliable.")


# --- Debug: Plot sum_of_windows (Still useful) ---
plt.figure(figsize=(12, 4))
time_axis_sumwin = np.arange(len(sum_of_windows)) / fs_recon * 1e6 # Time in us
plt.plot(time_axis_sumwin, sum_of_windows, label='Sum of Windows')
plt.title('Sum of Windows Across Reconstructed Signal')
plt.xlabel('Time (µs)')
plt.ylabel('Window Sum Magnitude')
min_sum = np.min(sum_of_windows)
max_sum = np.max(sum_of_windows)
# Set Y limits slightly beyond min/max for better visibility
plt.ylim(min(0, min_sum - 0.1*abs(min_sum)), max_sum + 0.1*abs(max_sum))
plt.grid(True)
plt.show()

# --- Normalization Step ---
print("\n--- Normalizing reconstructed signal using Sum of Windows ---")

print("Signal Stats BEFORE Normalization:")
rms_before_norm = np.nan
max_abs_before_norm = np.nan
if np.all(np.isfinite(reconstructed_signal)):
    rms_before_norm = np.sqrt(np.mean(np.real(reconstructed_signal * np.conj(reconstructed_signal))))
    max_abs_before_norm = np.max(np.abs(reconstructed_signal)) if len(reconstructed_signal) > 0 else 0
    print(f"  RMS: {rms_before_norm:.4e}")
    print(f"  Max Abs: {max_abs_before_norm:.4e}")
else:
    num_non_finite_before = np.sum(~np.isfinite(reconstructed_signal))
    print(f"  Contains {num_non_finite_before} non-finite values!")

# Determine a reliable fallback value for the divisor where sum_of_windows is too small
reliable_threshold = 1e-4 # Threshold below which sum_of_windows is considered unreliable
reliable_indices = np.where(sum_of_windows >= reliable_threshold)[0]
fallback_divisor = 1.0 # Default fallback if no reliable regions exist

if len(reliable_indices) > 0:
    # Use the median of the sum in reliable regions as a robust fallback
    median_reliable_sum = np.median(sum_of_windows[reliable_indices])
    if np.isfinite(median_reliable_sum) and median_reliable_sum > 1e-9: # Check if median is valid and non-zero
        fallback_divisor = median_reliable_sum
        print(f"Using median sum_of_windows in reliable regions ({median_reliable_sum:.4f}) as fallback divisor.")
    else:
        print(f"Warning: Median sum_of_windows ({median_reliable_sum}) in reliable regions is invalid or too small. Using fallback divisor {fallback_divisor}.")
else:
    print(f"Warning: No reliable regions found in sum_of_windows (threshold {reliable_threshold}). Using fallback divisor {fallback_divisor}.")

# Create the divisor array, replacing unreliable values
sum_of_windows_divisor = sum_of_windows.copy()
# Identify indices where sum is below threshold OR non-finite (safety check)
unreliable_indices = np.where((sum_of_windows < reliable_threshold) | (~np.isfinite(sum_of_windows)))[0]
sum_of_windows_divisor[unreliable_indices] = fallback_divisor
# Also check for exact zeros in the divisor that might have slipped through
zero_indices = np.where(np.abs(sum_of_windows_divisor) < 1e-15)[0]
if len(zero_indices) > 0:
    print(f"Warning: Found {len(zero_indices)} zero values in divisor after fallback. Replacing with fallback {fallback_divisor}.")
    sum_of_windows_divisor[zero_indices] = fallback_divisor # Replace zeros too


# Perform the normalization: signal = signal / divisor
# Use np.divide with out= and where= for potentially safer division
reconstructed_signal_normalized = np.zeros_like(reconstructed_signal, dtype=np.complex128)
valid_divisor_indices = np.abs(sum_of_windows_divisor) > 1e-15 # Where divisor is not effectively zero
np.divide(reconstructed_signal, sum_of_windows_divisor, out=reconstructed_signal_normalized, where=valid_divisor_indices)
# For indices where divisor was zero, the output remains zero (as initialized)

# Check for non-finite values AFTER normalization
normalization_successful = True
if not np.all(np.isfinite(reconstructed_signal_normalized)):
     num_non_finite_after = np.sum(~np.isfinite(reconstructed_signal_normalized))
     print(f"*** WARNING: {num_non_finite_after} non-finite values detected AFTER normalization! ***")
     # Option: Replace NaNs/Infs with 0?
     reconstructed_signal_normalized[~np.isfinite(reconstructed_signal_normalized)] = 0
     print("   Non-finite values replaced with 0.")
     normalization_successful = False # Mark failure


print("\nSignal Stats AFTER Normalization:")
rms_after_norm = np.nan
max_abs_after_norm = np.nan
if np.all(np.isfinite(reconstructed_signal_normalized)): # Check again after potential zeroing
    rms_after_norm = np.sqrt(np.mean(np.real(reconstructed_signal_normalized * np.conj(reconstructed_signal_normalized))))
    max_abs_after_norm = np.max(np.abs(reconstructed_signal_normalized)) if len(reconstructed_signal_normalized) > 0 else 0
    print(f"  RMS: {rms_after_norm:.4e}")
    print(f"  Max Abs: {max_abs_after_norm:.4e}")
    if EXPECTED_RMS > 1e-12:
         print(f"  Comparison to EXPECTED_RMS ({EXPECTED_RMS:.4e}): Ratio = {rms_after_norm/EXPECTED_RMS:.4f}")
    if not normalization_successful:
         print("  (Note: Normalization encountered issues.)")

else:
    # Should not happen if NaNs were zeroed, but include for completeness
    print("  Contains non-finite values even after attempted correction!")
    normalization_successful = False


# Update the main signal variable
reconstructed_signal = reconstructed_signal_normalized.copy()
print("\nNormalization complete.")


# --- 4. Evaluation & Visualization ---
print("\n--- Evaluating Reconstruction ---")

# --- Ground Truth Regeneration ---
print("Regenerating ground truth baseband for comparison...")
gt_duration = total_duration_recon # Use the same duration as reconstruction
num_samples_gt_compare = int(round(gt_duration * fs_recon))
# Regenerate time vector for GT
t_gt_compare = np.linspace(0, gt_duration, num_samples_gt_compare, endpoint=False)
gt_baseband_compare = np.zeros(num_samples_gt_compare, dtype=np.complex128) # Initialize

# Get parameters needed for GT generation from global attributes
mod = global_attrs.get('modulation', 'qam16') # Default to qam16
bw_gt = global_attrs.get('total_signal_bandwidth_hz', None) # Use the total BW attribute

if bw_gt is None:
     print("Error: Ground truth bandwidth ('total_signal_bandwidth_hz') not found in metadata. Cannot generate GT.")
     # Create dummy GT or exit
     gt_baseband_compare = (np.random.randn(num_samples_gt_compare) + 1j*np.random.randn(num_samples_gt_compare)) * EXPECTED_RMS / np.sqrt(2) # Noise
else:
    # --- QAM16 Ground Truth Generation ---
    if mod.lower() == 'qam16':
        # Assumption: For QAM, the total signal bandwidth might be related to symbol rate
        # A common assumption is Bandwidth = SymbolRate * (1 + RollOffFactor)
        # If RollOffFactor is unknown, a simple guess is SymbolRate approx Bandwidth
        # Let's use bw_gt as the symbol rate for this example. Adjust if simulation details differ.
        symbol_rate_gt = bw_gt
        print(f"Using GT Symbol Rate = Total Bandwidth = {symbol_rate_gt/1e6:.2f} Msps for {mod}")

        num_symbols_gt = int(np.ceil(gt_duration * symbol_rate_gt))
        if num_symbols_gt <= 0:
             print("Error: Calculated number of GT symbols is zero or negative.")
             # Handle error
        else:
            # Generate QAM16 symbols: points are (-3, -1, 1, 3) / sqrt(10) for unit power
            qam_points = [-3, -1, 1, 3]
            real_parts = np.random.choice(qam_points, size=num_symbols_gt)
            imag_parts = np.random.choice(qam_points, size=num_symbols_gt)
            symbols = (real_parts + 1j*imag_parts) / np.sqrt(10) # Avg Power = 1

            # Upsample symbols to match the reconstruction sample rate using zero-order hold
            samples_per_symbol_gt = fs_recon / symbol_rate_gt
            if samples_per_symbol_gt < 1:
                print(f"Warning: Samples per GT symbol ({samples_per_symbol_gt:.2f}) is less than 1. Check sample rates and bandwidth.")
                # Cannot properly represent symbols, GT will be inaccurate.

            # Create indices for repeating symbols
            indices = np.floor(np.arange(num_samples_gt_compare) / samples_per_symbol_gt).astype(int)
            # Ensure indices do not exceed the bounds of the generated symbols array
            indices = np.minimum(indices, num_symbols_gt - 1)
            indices = np.maximum(indices, 0) # Ensure non-negative indices

            gt_baseband_compare = symbols[indices]

            # --- Final GT Power Scaling ---
            # Scale the unit-power GT baseband to match the EXPECTED_RMS
            gt_scale_factor = EXPECTED_RMS
            gt_baseband_compare *= gt_scale_factor
            rms_gt = np.sqrt(np.mean(np.real(gt_baseband_compare*np.conj(gt_baseband_compare))))
            print(f"Scaled GT baseband. Target RMS: {EXPECTED_RMS:.4e}, Actual RMS: {rms_gt:.4e}")

    else:
        print(f"Warning: Ground Truth regeneration not implemented for modulation '{mod}'. Using noise.")
        gt_baseband_compare = (np.random.randn(num_samples_gt_compare) + 1j*np.random.randn(num_samples_gt_compare)) * EXPECTED_RMS / np.sqrt(2)

# --- Evaluation Metrics ---
# Use the same reliable indices definition as in the normalization step
reliable_indices_eval = np.where(sum_of_windows >= reliable_threshold)[0]

# Initialize metrics
mse = np.inf
nmse = np.inf
evm_percent = np.inf

if len(reliable_indices_eval) < 2:
    print("Evaluation Warning: Very few reliable samples (<2). Cannot calculate metrics reliably.")
    # Use the unaligned signal for plotting if alignment isn't possible
    reconstructed_signal_aligned = reconstructed_signal.copy()
else:
    print(f"Evaluating using {len(reliable_indices_eval)} reliable samples (where sum_of_windows >= {reliable_threshold}).")

    # Extract reliable portions of GT and Reconstruction
    # Ensure reliable indices are within the bounds of both signals
    max_index = min(len(gt_baseband_compare)-1, len(reconstructed_signal)-1)
    reliable_indices_eval = reliable_indices_eval[reliable_indices_eval <= max_index]

    if len(reliable_indices_eval) < 2:
         print("Evaluation Warning: After index bounds check, too few reliable samples remain (<2).")
         reconstructed_signal_aligned = reconstructed_signal.copy()
    else:
        gt_reliable = gt_baseband_compare[reliable_indices_eval]
        recon_reliable = reconstructed_signal[reliable_indices_eval] # Use the final (normalized) signal

        # Check again for non-finite values in the extracted reliable segments
        if not np.all(np.isfinite(gt_reliable)) or not np.all(np.isfinite(recon_reliable)):
            print("Evaluation Error: Non-finite values found in reliable segments AFTER extraction. Cannot calculate metrics.")
            reconstructed_signal_aligned = reconstructed_signal.copy() # Use unaligned for plot
        else:
            # --- Calculate Mean Squared Error (MSE) ---
            error_reliable = gt_reliable - recon_reliable
            mse = np.mean(np.real(error_reliable * np.conj(error_reliable))) # Equivalent to mean(abs(error)**2)

            # --- Calculate Normalized Mean Squared Error (NMSE) ---
            mean_power_gt_reliable = np.mean(np.real(gt_reliable * np.conj(gt_reliable)))
            if mean_power_gt_reliable > 1e-20: # Avoid division by zero
                 nmse = mse / mean_power_gt_reliable
            else:
                 nmse = np.inf # Undefined if GT power is zero

            # --- Calculate Error Vector Magnitude (EVM) ---
            # EVM (%) = sqrt( MSE / Mean Power of GT ) * 100 = sqrt(NMSE) * 100
            if np.isfinite(nmse) and nmse >= 0:
                evm_percent = np.sqrt(nmse) * 100
            else:
                evm_percent = np.inf

            # --- Alignment for Plotting ---
            # Scale the *entire* reconstructed signal so its power *in the reliable region* matches GT power *in the reliable region*
            reconstructed_signal_aligned = reconstructed_signal.copy() # Start with a copy
            mean_power_recon_reliable = np.mean(np.real(recon_reliable * np.conj(recon_reliable)))

            if mean_power_recon_reliable > 1e-20 and mean_power_gt_reliable > 1e-20:
                 # Calculate scale factor needed to match powers
                 power_scale_factor = np.sqrt(mean_power_gt_reliable / mean_power_recon_reliable)
                 # Apply scale factor to the *entire* signal for consistent plotting
                 reconstructed_signal_aligned *= power_scale_factor
                 print(f"Applied plotting alignment scale factor: {power_scale_factor:.4f}")
            else:
                 print("Warning: Cannot align powers for plotting due to zero power in reliable regions.")
                 # Keep reconstructed_signal_aligned as the unscaled version

print(f"\nEvaluation Metrics (on reliable samples):")
print(f"  MSE : {mse:.4e}")
if np.isfinite(nmse):
     print(f"  NMSE: {nmse:.4e} ({10*np.log10(nmse):.2f} dB)")
else:
     print(f"  NMSE: Infinite / Undefined")
print(f"  EVM : {evm_percent:.2f}%")


# --- Plotting ---
plt.style.use('seaborn-v0_8-darkgrid')

# --- Time Domain Plot (Matplotlib) ---
print("\n--- Generating Matplotlib Time Domain Plot ---")
fig_mpl, axs_mpl = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Determine plot length, ensuring it doesn't exceed available data in GT or (aligned) Recon
plot_samples = min(plot_length, len(t_gt_compare), len(reconstructed_signal_aligned))

if plot_samples <= 0:
     print("Warning: No samples available for plotting.")
else:
    time_axis_plot = t_gt_compare[:plot_samples] * 1e6 # Time axis in microseconds

    # --- Plot Ground Truth ---
    gt_plot_data = gt_baseband_compare[:plot_samples]
    # Replace any NaNs/Infs in plot data with 0 (shouldn't happen ideally, but safety)
    gt_plot_data_safe = np.nan_to_num(gt_plot_data)
    axs_mpl[0].plot(time_axis_plot, np.real(gt_plot_data_safe), label='GT (Real)')
    axs_mpl[0].plot(time_axis_plot, np.imag(gt_plot_data_safe), label='GT (Imag)', alpha=0.7)
    axs_mpl[0].set_title(f'Ground Truth (First {plot_samples} samples)')
    axs_mpl[0].set_ylabel('Amplitude')
    axs_mpl[0].legend(fontsize='small')
    axs_mpl[0].grid(True)
    # Determine sensible Y limits for GT
    max_abs_gt_plot = np.max(np.abs(gt_plot_data_safe)) if len(gt_plot_data_safe)>0 else 1.0
    ylim_gt = max(max_abs_gt_plot * 1.2, 0.05) # Add 20% margin, minimum range 0.1
    axs_mpl[0].set_ylim(-ylim_gt, ylim_gt)

    # --- Plot Reconstructed Signal (Aligned for Plotting) ---
    recon_plot_data = reconstructed_signal_aligned[:plot_samples]
    recon_plot_data_safe = np.nan_to_num(recon_plot_data) # Replace NaNs/Infs with 0 for plotting
    axs_mpl[1].plot(time_axis_plot, np.real(recon_plot_data_safe), label='Recon (Real)')
    axs_mpl[1].plot(time_axis_plot, np.imag(recon_plot_data_safe), label='Recon (Imag)', alpha=0.7)
    axs_mpl[1].set_title(f'Reconstructed (Aligned for Plot) - EVM: {evm_percent:.2f}%')
    axs_mpl[1].set_ylabel('Amplitude')
    axs_mpl[1].legend(fontsize='small')
    axs_mpl[1].grid(True)
    # Match Y limits to GT plot for direct comparison
    axs_mpl[1].set_ylim(axs_mpl[0].get_ylim())

    # --- Plot Error Signal ---
    # Ensure lengths match exactly for subtraction
    len_min_plot = min(len(gt_plot_data_safe), len(recon_plot_data_safe))
    error_signal = gt_plot_data_safe[:len_min_plot] - recon_plot_data_safe[:len_min_plot]
    error_signal_safe = np.nan_to_num(error_signal) # Safety check for error signal
    axs_mpl[2].plot(time_axis_plot[:len_min_plot], np.real(error_signal_safe), label='Error (Real)')
    axs_mpl[2].plot(time_axis_plot[:len_min_plot], np.imag(error_signal_safe), label='Error (Imag)', alpha=0.7)
    axs_mpl[2].set_title('Error (GT - Recon Aligned)')
    axs_mpl[2].set_xlabel('Time (µs)')
    axs_mpl[2].set_ylabel('Amplitude')
    axs_mpl[2].legend(fontsize='small')
    axs_mpl[2].grid(True)
    # Set error Y limits, perhaps smaller range than signal plots
    max_abs_err_plot = np.max(np.abs(error_signal_safe)) if len(error_signal_safe)>0 else 1.0
    ylim_err = max(max_abs_err_plot * 1.2, 0.02) # Add 20% margin, minimum range 0.04
    axs_mpl[2].set_ylim(-ylim_err, ylim_err)


    plt.tight_layout()
    plt.show()


# --- Spectrum Plot (Matplotlib) ---
print("\n--- Generating Matplotlib Spectrum Plot ---")
plt.figure(figsize=(12, 7))
plot_spectrum_flag = False # Flag to check if anything was plotted

# Plot Reconstructed Spectrum (using aligned signal for consistency with time plot)
if len(reconstructed_signal_aligned) > 1: # Need > 1 sample for FFT
    # Check finiteness before FFT
    if np.all(np.isfinite(reconstructed_signal_aligned)):
        f_recon, spec_recon_db = compute_spectrum(reconstructed_signal_aligned, fs_recon)
        if len(f_recon) > 0:
             plt.plot(f_recon/1e6, spec_recon_db, label='Recon Spec (Aligned)', ls='--', alpha=0.8)
             plot_spectrum_flag = True
    else:
        print("Skipping reconstructed spectrum plot: Aligned signal contains non-finite values.")
        # Option: Plot the non-aligned signal if it's finite?
        # if np.all(np.isfinite(reconstructed_signal)):
        #     f_recon, spec_recon_db = compute_spectrum(reconstructed_signal, fs_recon)
        #     if len(f_recon)>0: plt.plot(f_recon/1e6, spec_recon_db, label='Recon Spec (NOT Aligned)', ls=':', color='red', alpha=0.7); plot_spectrum_flag=True

else:
    print("Skipping reconstructed spectrum plot: Not enough samples.")

# Plot Ground Truth Spectrum
if len(gt_baseband_compare) > 1:
    if np.all(np.isfinite(gt_baseband_compare)):
        f_gt, spec_gt_db = compute_spectrum(gt_baseband_compare, fs_recon)
        if len(f_gt) > 0:
            plt.plot(f_gt/1e6, spec_gt_db, label='GT Spec', alpha=0.8, color='C0') # Use consistent color
            plot_spectrum_flag = True
    else:
        print("Skipping GT spectrum plot: GT signal contains non-finite values.")
else:
    print("Skipping GT spectrum plot: Not enough samples.")


if plot_spectrum_flag:
    plt.title('Spectra Comparison (Peak Normalized)')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Magnitude (dB)')
    plt.ylim(spectrum_ylim, 5) # Use parameter for lower limit, fixed upper limit
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No valid data plotted for spectrum comparison.")


# --- Time Domain Plot (Plotly) ---
print("\n--- Generating Plotly Time Domain Plot for Comparison ---")
if plot_samples > 0:
    fig_plotly = go.Figure()

    # Add Ground Truth Traces (using the same 'safe' data as matplotlib)
    fig_plotly.add_trace(go.Scatter(x=time_axis_plot, y=np.real(gt_plot_data_safe),
                                     mode='lines', name='GT (Real)',
                                     line=dict(color='blue')))
    fig_plotly.add_trace(go.Scatter(x=time_axis_plot, y=np.imag(gt_plot_data_safe),
                                     mode='lines', name='GT (Imag)',
                                     line=dict(color='lightblue', dash='dash')))

    # Add Reconstructed Traces (using the same 'safe' aligned data as matplotlib)
    fig_plotly.add_trace(go.Scatter(x=time_axis_plot, y=np.real(recon_plot_data_safe),
                                     mode='lines', name='Recon (Real)',
                                     line=dict(color='red')))
    fig_plotly.add_trace(go.Scatter(x=time_axis_plot, y=np.imag(recon_plot_data_safe),
                                     mode='lines', name='Recon (Imag)',
                                     line=dict(color='orange', dash='dash')))

    fig_plotly.update_layout(
        title=f'Plotly Time Domain Comparison (First {plot_samples} samples) - Hover to see values',
        xaxis_title='Time (µs)',
        yaxis_title='Amplitude',
        legend_title='Signal',
        hovermode='x unified' # Show hover info for all traces at a given x
    )

    # --- Compare Y-Axis Range ---
    # Get Matplotlib Y-axis limits from the Recon plot for reference
    try:
        mpl_ylim_recon = axs_mpl[1].get_ylim()
        print(f"Reference: Matplotlib Recon Y-Axis Limits: ({mpl_ylim_recon[0]:.4f}, {mpl_ylim_recon[1]:.4f})")
        # Optionally force Plotly's Y-axis to match Matplotlib's to see if it *can* display that range
        # fig_plotly.update_yaxes(range=mpl_ylim_recon)
    except Exception as e:
        print(f"Could not get Matplotlib Y-limits for comparison: {e}")


    # Let Plotly auto-scale by default and compare visually/by hovering.
    fig_plotly.show()

else:
    print("Skipping Plotly plot due to no samples available.")


print("\nScript finished.")