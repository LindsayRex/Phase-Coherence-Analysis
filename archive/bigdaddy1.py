import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import h5py
from tqdm import tqdm
import pywt  # For Wavelets
import tensorly as tl  # For Tensor Decomposition (CP)
from tensorly.decomposition import parafac
import sys # For exiting on critical errors
import plotly.graph_objects as go # For comparison plot

# --- Tuning Parameters ---
# (Keep these as defined in your original "Big Daddy" script)
filter_length = 32
mu = 0.0001
cp_rank = 5
num_lags = 3
delay_samples = 1
cp_iter_max = 200
cp_tolerance = 1e-6
wavelet_name = 'db4'
wpd_level = 4
stitching_window_type = 'hann'
plot_length = 5000
spectrum_ylim = -100
input_filename = "simulated_chunks_25GHz_400MHzBW_qam16_sdr56MHz.h5"
EXPECTED_RMS_AFTER_SCALING = 1.39e-02 # Define the target RMS after initial scaling


# --- Helper function for spectrum calculation ---
def compute_spectrum(signal_data, sample_rate):
    """Calculates the FFT spectrum, shifts it, converts to dB, and normalizes."""
    n = len(signal_data)
    if n < 2: return np.array([]), np.array([])
    # Ensure input is finite before FFT
    if not np.all(np.isfinite(signal_data)):
        print("Warning: Non-finite data passed to compute_spectrum. Returning empty.")
        return np.array([]), np.array([])
    try:
        f = np.fft.fftshift(np.fft.fftfreq(n, d=1/sample_rate))
        s = np.fft.fftshift(np.fft.fft(signal_data))
        spec_mag = np.abs(s)
        spec_db = 20 * np.log10(spec_mag + 1e-12)
        max_db = np.nanmax(spec_db)
        if np.isfinite(max_db): spec_db -= max_db
        else: print("Warning: Max spectrum value non-finite. Not normalizing.")
        return f, spec_db
    except Exception as e:
        print(f"Error in compute_spectrum: {e}")
        return np.array([]), np.array([])

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
        if actual_chunks == 0: raise ValueError("No chunks found in HDF5 file.")
        for i in range(actual_chunks):
            chunk_name = f'chunk_{i:03d}'
            if chunk_name in f:
                group = f[chunk_name]
                # Ensure correct data type on load
                chunk_data = group['iq_data'][:].astype(np.complex128)
                meta = {key: value for key, value in group.attrs.items()}
                loaded_chunks.append(chunk_data)
                loaded_metadata.append(meta)
            else: print(f"Warning: Chunk {chunk_name} not found.")
except Exception as e:
    print(f"Error loading HDF5 file '{input_filename}': {e}"); sys.exit(1)
if not loaded_chunks: print("No chunk data loaded. Exiting."); sys.exit(1)
print(f"\nSuccessfully loaded {len(loaded_chunks)} chunks.")

fs_sdr = global_attrs.get('sdr_sample_rate_hz', None)
fs_recon = global_attrs.get('ground_truth_sample_rate_hz', None)
if fs_sdr is None or fs_recon is None:
    print("Error: Sample rate information missing in HDF5 attributes."); sys.exit(1)


# --- 1b. Correct Initial Amplitude Scaling ---
print("\n--- Correcting Initial Amplitude Scaling ---")
scaled_loaded_chunks = []
print("Chunk | RMS Before | Scaling Factor | RMS After  | Max Abs (After)")
print("------|------------|----------------|------------|----------------")
overall_scaling_successful = True
for i, chunk in enumerate(loaded_chunks):
    if len(chunk) == 0:
        scaled_loaded_chunks.append(chunk)
        print(f"{i:<5d} | --- EMPTY -- | ---            | --- EMPTY -- | ---")
        continue
    if not np.all(np.isfinite(chunk)):
        print(f"ERROR: Chunk {i} non-finite BEFORE scaling.")
        overall_scaling_successful = False
        scaled_loaded_chunks.append(chunk) # Keep original
        continue
    rms_before = np.sqrt(np.mean(np.real(chunk * np.conj(chunk))))
    if rms_before < 1e-12:
        scaling_factor = 0.0 # Or 1.0? If RMS is 0, scaling doesn't matter
        scaled_chunk = chunk # Keep as is
        print(f"{i:<5d} | {rms_before:.4e} | SKIPPED (Zero) | {rms_before:.4e} | {np.max(np.abs(chunk)):.4e}")
    else:
        scaling_factor = EXPECTED_RMS_AFTER_SCALING / rms_before
        scaled_chunk = (chunk * scaling_factor).astype(np.complex128)

    if not np.all(np.isfinite(scaled_chunk)):
         print(f"ERROR: Chunk {i} non-finite AFTER scaling (Factor: {scaling_factor:.4f}). RMS before: {rms_before:.4e}")
         overall_scaling_successful = False
         # Keep the chunk with NaNs for potential inspection
         scaled_loaded_chunks.append(scaled_chunk)
         continue

    rms_after = np.sqrt(np.mean(np.real(scaled_chunk * np.conj(scaled_chunk))))
    max_abs_after = np.max(np.abs(scaled_chunk)) if len(scaled_chunk)>0 else 0
    print(f"{i:<5d} | {rms_before:.4e} | {scaling_factor:<14.4f} | {rms_after:.4e} | {max_abs_after:.4e}")
    scaled_loaded_chunks.append(scaled_chunk)

if not overall_scaling_successful:
    print("\nERROR: Non-finite values detected during initial scaling. Proceeding cautiously.")
    # Consider exiting: sys.exit(1)

# Use scaled chunks for the next step
original_chunks_for_af = scaled_loaded_chunks


# --- 2. Adaptive Filtering Pre-Alignment ---
print("\n--- SKIPPING Adaptive Filtering Pre-Alignment (FOR DEBUGGING) ---")
#print("\n--- Performing Adaptive Filtering Pre-Alignment ---")

"""code off for debugg

aligned_chunks_af = [] # Store results of AF
af_successful = True
if len(original_chunks_for_af) > 0:
    aligned_chunks_af.append(original_chunks_for_af[0].copy()) # Start with the first chunk (no alignment needed)
else:
    print("Error: No scaled chunks available for AF.")
    sys.exit(1)


for i in tqdm(range(1, len(original_chunks_for_af)), desc="AF Aligning chunks"):
    if i >= len(loaded_metadata): # Ensure metadata index is valid
         print(f"Warning: Metadata index {i} out of bounds. Skipping AF for this chunk.")
         if i < len(original_chunks_for_af): aligned_chunks_af.append(original_chunks_for_af[i].copy())
         continue

    prev_chunk = original_chunks_for_af[i - 1]
    curr_chunk = original_chunks_for_af[i]
    meta_prev = loaded_metadata[i - 1]
    meta_curr = loaded_metadata[i]

    # Check for empty chunks
    if len(prev_chunk) == 0 or len(curr_chunk) == 0:
         print(f"Warning: Chunk {i-1} or {i} is empty. Skipping AF.")
         aligned_chunks_af.append(curr_chunk.copy())
         continue

    overlap_factor = global_attrs.get('overlap_factor', 0.1)
    chunk_samples = len(curr_chunk)
    overlap_samples = int(chunk_samples * overlap_factor)

    # Ensure overlap is valid and sufficient
    if overlap_samples <= filter_length or overlap_samples > len(prev_chunk) or overlap_samples > len(curr_chunk):
        print(f"Warning: Chunk {i} overlap invalid/insufficient (Overlap:{overlap_samples}, Filter:{filter_length}, PrevLen:{len(prev_chunk)}, CurrLen:{len(curr_chunk)}). Skipping AF.")
        aligned_chunks_af.append(curr_chunk.copy())
        continue

    ref_signal = prev_chunk[-overlap_samples:]
    input_signal = curr_chunk[:overlap_samples]

    # Check for non-finite values before AF
    if not np.all(np.isfinite(ref_signal)) or not np.all(np.isfinite(input_signal)):
         print(f"Warning: Non-finite values in overlap region for chunk {i}. Skipping AF.")
         aligned_chunks_af.append(curr_chunk.copy())
         continue

    # Coarse delay estimation (optional but can help LMS)
    # corr = sig.correlate(ref_signal, input_signal, mode='full')
    # delay_est = np.argmax(np.abs(corr)) - (len(input_signal) - 1)
    # input_signal_shifted = np.roll(input_signal, delay_est) # Apply coarse shift if needed
    # Use input_signal directly if not shifting
    input_signal_lms = input_signal

    # LMS adaptation
    weights = np.zeros(filter_length, dtype=complex)
    y = np.zeros(len(input_signal_lms), dtype=complex) # Output buffer
    lms_converged = True
    try:
        for n in range(filter_length, len(input_signal_lms)):
            x_n = input_signal_lms[n - filter_length:n][::-1] # Input vector (reversed)
            y[n] = np.dot(weights, x_n) # Filter output
            error = ref_signal[n] - y[n] # Error calculation
            weights += mu * np.conj(error) * x_n # Standard LMS update (using conj(error)) - Check literature if conj(x_n) preferred
            # Check for weight explosion
            if not np.all(np.isfinite(weights)):
                 print(f"Warning: Non-finite weights encountered in LMS for chunk {i} at step {n}. Resetting weights.")
                 weights = np.zeros(filter_length, dtype=complex) # Reset
                 lms_converged = False
                 break # Stop LMS for this chunk
    except Exception as lms_e:
        print(f"Error during LMS computation for chunk {i}: {lms_e}")
        weights = np.zeros(filter_length, dtype=complex) # Use zero weights on error
        lms_converged = False
        af_successful = False

    # Apply filter to entire current chunk only if LMS seemed okay
    if lms_converged and np.all(np.isfinite(weights)):
        # Use 'full' convolution and trim, or 'same' if confident about boundaries
        # 'full' is safer: output len = N + M - 1
        filtered_full = sig.convolve(curr_chunk, weights, mode='full')
        # Trim to original length (adjust based on filter centering, usually take central part)
        start_trim = (filter_length - 1) // 2
        end_trim = start_trim + len(curr_chunk)
        if end_trim <= len(filtered_full):
             aligned_chunk_conv = filtered_full[start_trim:end_trim]
             aligned_chunks_af.append(aligned_chunk_conv)
             # print(f"Chunk {i}: Applied AF. Max weight mag={np.max(np.abs(weights)):.4f}") # Optional debug
        else:
             print(f"Warning: AF convolution trimming error for chunk {i}. Using original.")
             aligned_chunks_af.append(curr_chunk.copy())
             af_successful = False
    else:
        print(f"Warning: Using original chunk {i} due to LMS issues.")
        aligned_chunks_af.append(curr_chunk.copy()) # Append original if LMS failed/weights bad
        if not lms_converged: af_successful = False


if not af_successful:
     print("Warning: Issues encountered during Adaptive Filtering.")

"""  # End of the multi-line string literal comment

# Directly use the scaled chunks as input for the next step (THIS LINE IS *NOT* COMMENTED)
aligned_chunks_af = original_chunks_for_af
af_successful = True # Assume success since skipped

# --- 3. Sub-Nyquist CP Decomposition ---
print("\n--- Performing CP Decomposition ---")
estimated_chunk_phases = []
estimated_chunk_dominant_freqs = []
true_chunk_phases_from_meta = [] # Store true phases for comparison
cp_successful = True

tl.set_backend('numpy') # Ensure numpy backend

for i, chunk_data in tqdm(enumerate(aligned_chunks_af), desc="CP Decomposition"):
    # Ensure metadata is available
    if i >= len(loaded_metadata):
        print(f"Warning: Metadata missing for chunk {i}. Cannot get true phase. Skipping CP.")
        estimated_chunk_phases.append(0.0) # Append dummy value
        estimated_chunk_dominant_freqs.append(0.0)
        true_chunk_phases_from_meta.append(np.nan)
        continue

    meta = loaded_metadata[i]
    # Store true phase if available, otherwise NaN
    true_applied_phase = meta.get('applied_phase_offset_rad', np.nan)
    true_chunk_phases_from_meta.append(true_applied_phase)

    # Check chunk length and finite values
    if len(chunk_data) < num_lags + delay_samples + cp_rank * 2: # Need enough samples
        print(f"  Skipping CP on chunk {i}: Not enough samples ({len(chunk_data)}).")
        estimated_chunk_phases.append(0.0) # Fallback estimate
        estimated_chunk_dominant_freqs.append(0.0)
        cp_successful = False
        continue
    if not np.all(np.isfinite(chunk_data)):
         print(f"ERROR: Non-finite values in input chunk {i} for CP. Skipping.")
         estimated_chunk_phases.append(0.0) # Fallback estimate
         estimated_chunk_dominant_freqs.append(0.0)
         cp_successful = False
         continue


    # Simulate delayed path
    chunk_delayed = np.roll(chunk_data, delay_samples)
    chunk_delayed[:delay_samples] = 0 # Zero out rolled samples

    # Build correlation tensor
    max_len = len(chunk_data) - num_lags - delay_samples # Max length for correlation products
    tensor_data = np.zeros((max_len, 2, num_lags), dtype=complex)
    try:
        for l in range(num_lags):
            # Autocorrelation for original and delayed paths
            tensor_data[:, 0, l] = chunk_data[l:max_len+l] * np.conj(chunk_data[:max_len])
            tensor_data[:, 1, l] = chunk_delayed[l:max_len+l] * np.conj(chunk_delayed[:max_len])
        if not np.all(np.isfinite(tensor_data)):
            raise ValueError("Non-finite values encountered during tensor construction.")
    except Exception as e:
        print(f"  Error building tensor for chunk {i}: {e}")
        estimated_chunk_phases.append(0.0) # Fallback estimate
        estimated_chunk_dominant_freqs.append(0.0)
        cp_successful = False
        continue

    # Perform CP Decomposition
    try:
        weights, factors = parafac(tensor_data, rank=cp_rank, init='random', tol=cp_tolerance, n_iter_max=cp_iter_max, verbose=0)

        # Check factors for finiteness
        if not all(np.all(np.isfinite(f)) for f in factors) or not np.all(np.isfinite(weights)):
             print(f"Warning: Non-finite values in CP factors/weights for chunk {i}. Skipping phase est.")
             estimated_chunk_phases.append(0.0)
             estimated_chunk_dominant_freqs.append(0.0)
             cp_successful = False
             continue

        # Find dominant component based on weights (energy)
        dominant_comp_idx = np.argmax(weights)
        lag_factor = factors[2][:, dominant_comp_idx] # Factor corresponding to lags dimension

        # Estimate phase slope (frequency) and initial phase
        valid_indices = np.where(np.abs(lag_factor) > 1e-6)[0] # Indices where factor is non-negligible
        if len(valid_indices) > 1:
            # Calculate phase differences between consecutive valid lags
            phase_diffs = np.angle(lag_factor[valid_indices[1:]] / lag_factor[valid_indices[:-1]])
            # Wrap angles to [-pi, pi]
            phase_diffs_wrapped = np.angle(np.exp(1j*phase_diffs))
            # Use median for robustness against outliers
            dominant_freq_rad_per_lag = np.median(phase_diffs_wrapped)
            # Convert rad/lag to Hz: freq_hz = (rad_per_lag / (2*pi)) * fs_sdr
            dominant_freq_hz = dominant_freq_rad_per_lag * fs_sdr / (2 * np.pi)
            # Estimate initial phase from the first valid factor
            phase_est = np.angle(lag_factor[valid_indices[0]])
        elif len(valid_indices) == 1:
             print(f"  Warning: Only one valid lag factor for chunk {i}. Cannot estimate frequency. Using phase only.")
             dominant_freq_hz = 0.0
             phase_est = np.angle(lag_factor[valid_indices[0]])
        else:
            print(f"  Warning: No valid lag factors for chunk {i}. Cannot estimate phase/frequency.")
            dominant_freq_hz = 0.0
            phase_est = 0.0 # Fallback estimate
            cp_successful = False


        # --- Debugging Output ---
        # print(f"Chunk {i}: CP Est Freq Offset={dominant_freq_hz/1e3:.1f} kHz, Phase={np.rad2deg(phase_est):.1f} deg")
        # if np.isfinite(true_applied_phase):
        #     print(f"         True Phase={np.rad2deg(true_applied_phase):.1f} deg, Error={np.rad2deg(phase_est - true_applied_phase):.1f} deg")
        # else:
        #     print(f"         True Phase= N/A")
        # print(f"         Weight Magnitudes: {[f'{abs(w):.2e}' for w in weights]}")
        # --- End Debugging ---

        estimated_chunk_phases.append(phase_est)
        estimated_chunk_dominant_freqs.append(dominant_freq_hz)

    except Exception as e:
        print(f"  CP decomposition failed for chunk {i}: {e}")
        estimated_chunk_phases.append(0.0) # Fallback estimate
        estimated_chunk_dominant_freqs.append(0.0)
        cp_successful = False
        true_chunk_phases_from_meta[i] = np.nan # Ensure consistency if CP fails


print("\nCP decomposition complete.")
if not cp_successful:
    print("Warning: Issues encountered during CP Decomposition. Phase estimates may be inaccurate.")

# --- Plot CP Phase Estimation vs True Phase ---
plt.figure(figsize=(10, 5))
chunk_indices = np.arange(len(estimated_chunk_phases))
valid_true_phases = np.isfinite(true_chunk_phases_from_meta)
if np.any(valid_true_phases):
    plt.plot(chunk_indices[valid_true_phases], np.rad2deg(np.array(true_chunk_phases_from_meta)[valid_true_phases]),
             'bo-', label='True Phase (from Meta)')
plt.plot(chunk_indices, np.rad2deg(np.unwrap(estimated_chunk_phases)), 'rx--', label='Estimated Phase (CP, unwrapped)') # Unwrap for clearer trend
plt.title('CP Phase Estimation vs. True Phase')
plt.xlabel('Chunk Index')
plt.ylabel('Phase (degrees)')
plt.legend()
plt.grid(True)
plt.show()

# --- 4. Upsample Aligned Chunks ---
print("\n--- Upsampling chunks (using FFT method) ---")
upsampled_chunks = []
debug_rms_upsample_input = []
debug_rms_upsample_output = []
debug_max_abs_upsample_input = []
debug_max_abs_upsample_output = []
upsampling_successful = True

for i, chunk_data in tqdm(enumerate(aligned_chunks_af), total=len(aligned_chunks_af), desc="Upsampling"):
    # Calculate RMS/Max before
    rms_in = np.nan
    max_abs_in = np.nan
    if len(chunk_data)>0 and np.all(np.isfinite(chunk_data)):
        rms_in = np.sqrt(np.mean(np.real(chunk_data * np.conj(chunk_data))))
        max_abs_in = np.max(np.abs(chunk_data))
    debug_rms_upsample_input.append(rms_in)
    debug_max_abs_upsample_input.append(max_abs_in)

    # Determine target number of samples
    chunk_duration_sdr = len(chunk_data) / fs_sdr if fs_sdr > 0 else 0
    num_samples_chunk_recon = int(round(chunk_duration_sdr * fs_recon))

    if len(chunk_data) < 2:
        print(f"Warning: Chunk {i} too short ({len(chunk_data)}) for resampling. Appending zeros.")
        upsampled_chunks.append(np.zeros(num_samples_chunk_recon, dtype=complex))
        debug_rms_upsample_output.append(0.0)
        debug_max_abs_upsample_output.append(0.0)
        continue
    if not np.isfinite(rms_in): # Skip if input is bad
        print(f"Warning: Non-finite input RMS for chunk {i}. Appending zeros.")
        upsampled_chunks.append(np.zeros(num_samples_chunk_recon, dtype=complex))
        debug_rms_upsample_output.append(np.nan)
        debug_max_abs_upsample_output.append(np.nan)
        upsampling_successful = False
        continue

    try:
        # sig.resample uses FFT, potentially faster for large ratios but different edge effects
        upsampled_chunk = sig.resample(chunk_data, num_samples_chunk_recon)
        upsampled_chunk = upsampled_chunk.astype(np.complex128) # Ensure type

        # Calculate RMS/Max after
        rms_out = np.nan
        max_abs_out = np.nan
        if np.all(np.isfinite(upsampled_chunk)):
            rms_out = np.sqrt(np.mean(np.real(upsampled_chunk * np.conj(upsampled_chunk))))
            max_abs_out = np.max(np.abs(upsampled_chunk))
        else:
            print(f"ERROR: Non-finite values after sig.resample for chunk {i}.")
            upsampling_successful = False
            # Replace with zeros?
            upsampled_chunk = np.zeros(num_samples_chunk_recon, dtype=complex)
            rms_out = 0.0
            max_abs_out = 0.0

        debug_rms_upsample_output.append(rms_out)
        debug_max_abs_upsample_output.append(max_abs_out)
        upsampled_chunks.append(upsampled_chunk.copy())

        # Plot first chunk comparison
        if i == 0:
            print("Plotting first upsampled chunk (FFT method)...")
            plt.figure(figsize=(12,4))
            time_axis_debug = np.arange(len(upsampled_chunk))/fs_recon*1e6
            plt.plot(time_axis_debug, np.real(upsampled_chunk), label='Real')
            plt.plot(time_axis_debug, np.imag(upsampled_chunk), label='Imag', alpha=0.7)
            plt.title(f'First Upsampled Chunk (RMS={rms_out:.3e}, MaxAbs={max_abs_out:.3e})')
            plt.xlabel('Time (µs)'); plt.ylabel('Amplitude'); plt.legend(); plt.grid(True)
            plot_duration_us = 5.0
            max_time_us = len(upsampled_chunk) / fs_recon * 1e6
            xlim_upper = min(plot_duration_us, max_time_us if max_time_us > 0 else plot_duration_us)
            plt.xlim(-0.1 * xlim_upper, xlim_upper)
            ylim_abs = max(max_abs_out * 1.2 if np.isfinite(max_abs_out) else 0.1, 0.05)
            plt.ylim(-ylim_abs, ylim_abs)
            plt.show()

            print("Plotting Spectra Comparison (After Upsampling)...")
            plt.figure(figsize=(12, 6))
            f_before, spec_before_db = compute_spectrum(chunk_data, fs_sdr)
            if len(f_before) > 0: plt.plot(f_before/1e6, spec_before_db, label=f'BEFORE (Fs={fs_sdr/1e6:.1f}MHz)', alpha=0.8)
            f_after, spec_after_db = compute_spectrum(upsampled_chunk, fs_recon)
            if len(f_after) > 0: plt.plot(f_after/1e6, spec_after_db, label=f'AFTER (Fs={fs_recon/1e6:.1f}MHz)', ls='--', alpha=0.8)
            plt.title('Spectra Comparison Upsampling (FFT Method)'); plt.xlabel('Frequency (MHz)'); plt.ylabel('Magnitude (dB)')
            plt.ylim(-120, 5); plt.legend(); plt.grid(True); plt.show()

    except Exception as resample_e:
        print(f"Error using sig.resample for chunk {i}: {resample_e}. Appending zeros.")
        upsampled_chunks.append(np.zeros(num_samples_chunk_recon, dtype=complex))
        debug_rms_upsample_output.append(np.nan)
        debug_max_abs_upsample_output.append(np.nan)
        upsampling_successful = False

if not upsampling_successful:
    print("\nWARNING: Issues encountered during upsampling.")

# Print Upsampling Summary Table
print("\n--- RMS and Max Abs Before/After Upsampling (FFT Method) ---")
print("Chunk | RMS Before | Max Abs Before | RMS After  | Max Abs After")
print("------|------------|----------------|------------|--------------")
min_len_debug = min(len(debug_rms_upsample_input), len(debug_max_abs_upsample_input),
                    len(debug_rms_upsample_output), len(debug_max_abs_upsample_output))
for i in range(min_len_debug):
     rms_in_str = f"{debug_rms_upsample_input[i]:.4e}" if np.isfinite(debug_rms_upsample_input[i]) else "  NaN     "
     max_in_str = f"{debug_max_abs_upsample_input[i]:.4e}" if np.isfinite(debug_max_abs_upsample_input[i]) else "  NaN     "
     rms_out_str = f"{debug_rms_upsample_output[i]:.4e}" if np.isfinite(debug_rms_upsample_output[i]) else "  NaN     "
     max_out_str = f"{debug_max_abs_upsample_output[i]:.4e}" if np.isfinite(debug_max_abs_upsample_output[i]) else "  NaN     "
     print(f"{i:<5d} | {rms_in_str} | {max_in_str}   | {rms_out_str} | {max_out_str}")


# --- 5. Wavelet-Based Processing AT HIGH RATE ---
print("\n--- Performing Wavelet Processing AT RECONSTRUCTION RATE ---")

# Calculate total duration and samples needed for reconstruction buffer
# Using intended duration from metadata if available, else estimate from samples
# Add tuning delay between chunks
chunk_duration_sdr_s = len(loaded_chunks[0]) / fs_sdr if len(loaded_chunks)>0 else 0 # Duration of one chunk at SDR rate
overlap_factor = global_attrs.get('overlap_factor', 0.1)
tuning_delay = global_attrs.get('tuning_delay_s', 5e-6)
time_advance_per_chunk = chunk_duration_sdr_s * (1.0 - overlap_factor) + tuning_delay
# Total duration needs duration of first chunk (at recon rate) + (N-1) advances
chunk_duration_recon_s = len(upsampled_chunks[0]) / fs_recon if len(upsampled_chunks)>0 else 0
total_duration_recon_est = chunk_duration_recon_s + (len(loaded_chunks) - 1) * time_advance_per_chunk
num_samples_recon = int(round(total_duration_recon_est * fs_recon))

if num_samples_recon <= 0:
     print(f"Error: Estimated reconstruction samples ({num_samples_recon}) is non-positive. Exiting.")
     sys.exit(1)

reconstructed_signal = np.zeros(num_samples_recon, dtype=complex)
sum_of_windows = np.zeros(num_samples_recon, dtype=float) # Use float for sum

print(f"Reconstruction target buffer: {num_samples_recon} samples @ {fs_recon/1e6:.2f} MHz (Est. Duration: {total_duration_recon_est*1e6:.1f} us)")

wpd_successful = True
current_recon_time_start = 0.0 # Track start time for stitching

# Check if wavelet exists
try:
    wavelet_obj = pywt.Wavelet(wavelet_name)
except ValueError:
    print(f"Error: Wavelet '{wavelet_name}' not found in PyWavelets. Exiting.")
    sys.exit(1)

for i, up_chunk in tqdm(enumerate(upsampled_chunks), total=len(upsampled_chunks), desc="Wavelet Recon"):
    # Ensure CP phase estimate is available
    if i >= len(estimated_chunk_phases):
         print(f"Warning: CP Phase estimate missing for chunk {i}. Skipping WPD.")
         wpd_successful = False
         # Need to advance time correctly here based on original chunk duration
         if i < len(upsampled_chunks) - 1: current_recon_time_start += time_advance_per_chunk
         continue

    cp_phase_estimate = estimated_chunk_phases[i] # Get phase from CP step

    # Check chunk validity and minimum length for WPD
    min_len_wpd = wavelet_obj.dec_len * (2**wpd_level) # Rough estimate
    if len(up_chunk) < min_len_wpd or not np.all(np.isfinite(up_chunk)):
        print(f"Warning: Upsampled chunk {i} too short ({len(up_chunk)} vs min {min_len_wpd}) or non-finite for WPD level {wpd_level}. Skipping.")
        wpd_successful = False
        if i < len(upsampled_chunks) - 1: current_recon_time_start += time_advance_per_chunk
        continue

    # --- Debug: RMS before WPD ---
    rms_before_wpd = np.sqrt(np.mean(np.real(up_chunk * np.conj(up_chunk))))

    try:
        # Perform WPD on the upsampled chunk
        wp = pywt.WaveletPacket(data=up_chunk, wavelet=wavelet_name, mode='symmetric', maxlevel=wpd_level)

        # Apply phase correction to leaf nodes at the desired level
        nodes = wp.get_level(wpd_level, order='natural') # Get nodes at the max level
        correction_applied = False
        for node in nodes:
            if node.level == wpd_level and node.data is not None: # Check level and data existence
                 # Apply phase correction (CP estimate)
                 node.data = node.data * np.exp(-1j * cp_phase_estimate)
                 correction_applied = True

        if not correction_applied:
            print(f"Warning: No leaf nodes found or corrected at level {wpd_level} for chunk {i}.")
            # If no correction applied, should we still reconstruct? Let's try.

        # Reconstruct the phase-corrected chunk
        reconstructed_chunk_wpd = wp.reconstruct(update=False) # Use modified coefficients

        # Check length and finite values after reconstruction
        if len(reconstructed_chunk_wpd) != len(up_chunk):
             print(f"Warning: WPD reconstructed chunk {i} length ({len(reconstructed_chunk_wpd)}) differs from upsampled ({len(up_chunk)}). Adjusting.")
             # Trim or pad to match original upsampled length for consistent stitching
             target_len = len(up_chunk)
             if len(reconstructed_chunk_wpd) > target_len:
                 reconstructed_chunk_wpd = reconstructed_chunk_wpd[:target_len]
             else:
                 reconstructed_chunk_wpd = np.pad(reconstructed_chunk_wpd, (0, target_len - len(reconstructed_chunk_wpd)), mode='constant')

        if not np.all(np.isfinite(reconstructed_chunk_wpd)):
             print(f"ERROR: Non-finite values after WPD reconstruction for chunk {i}. Replacing with zeros.")
             reconstructed_chunk_wpd = np.zeros_like(up_chunk)
             wpd_successful = False

    except Exception as wp_e:
        print(f"Error during Wavelet processing for chunk {i}: {wp_e}. Replacing with zeros.")
        reconstructed_chunk_wpd = np.zeros_like(up_chunk) # Use zeros if WPD fails
        wpd_successful = False


    # --- Debug: RMS after WPD ---
    rms_after_wpd = np.sqrt(np.mean(np.real(reconstructed_chunk_wpd * np.conj(reconstructed_chunk_wpd))))
    if i == 0: # Print comparison only for the first chunk
        print(f"Chunk 0 RMS: Before WPD={rms_before_wpd:.4e}, After WPD={rms_after_wpd:.4e}")


    # --- Stitching (Overlap-Add) ---
    start_idx_recon = int(round(current_recon_time_start * fs_recon))
    num_samples_in_chunk = len(reconstructed_chunk_wpd)
    end_idx_recon = start_idx_recon + num_samples_in_chunk

    # Boundary checks for stitching
    if start_idx_recon < 0:
        print(f"Warning: Stitching start index negative ({start_idx_recon}) for chunk {i}. Skipping.")
        if i < len(upsampled_chunks) - 1: current_recon_time_start += time_advance_per_chunk
        continue
    if start_idx_recon >= num_samples_recon: # Started past the end of the buffer
        print(f"Warning: Stitching start index ({start_idx_recon}) exceeds buffer length ({num_samples_recon}) for chunk {i}. Stopping stitch.")
        break # No point continuing if we are past the end

    # Truncate chunk if it extends beyond buffer
    actual_len_to_add = num_samples_in_chunk
    if end_idx_recon > num_samples_recon:
        actual_len_to_add = num_samples_recon - start_idx_recon
        end_idx_recon = num_samples_recon # Adjust end index

    if actual_len_to_add <= 0: # Skip if nothing to add
        if i < len(upsampled_chunks) - 1: current_recon_time_start += time_advance_per_chunk
        continue

    # Get window
    if actual_len_to_add < 2: window = np.ones(actual_len_to_add)
    else: window = sig.get_window(stitching_window_type, actual_len_to_add)

    # Add to buffers
    try:
        segment_to_add = reconstructed_chunk_wpd[:actual_len_to_add] * window
        if not np.all(np.isfinite(segment_to_add)):
             print(f"Warning: Non-finite values in segment_to_add for chunk {i}. Zeroing.")
             segment_to_add[~np.isfinite(segment_to_add)] = 0
             wpd_successful = False

        reconstructed_signal[start_idx_recon:end_idx_recon] += segment_to_add
        sum_of_windows[start_idx_recon:end_idx_recon] += window

        # Check for non-finite result in buffer (overflow?)
        if not np.all(np.isfinite(reconstructed_signal[start_idx_recon:end_idx_recon])):
             print(f"CRITICAL WARNING: Non-finite values in recon buffer after adding chunk {i}. Zeroing segment.")
             reconstructed_signal[start_idx_recon:end_idx_recon] = 0
             wpd_successful = False

    except IndexError as idx_e:
        print(f"Error (IndexError) during stitching chunk {i}: {idx_e}")
        print(f"  Indices: start={start_idx_recon}, end={end_idx_recon}, actual_len={actual_len_to_add}, recon_len={num_samples_recon}")
        wpd_successful = False
    except Exception as stitch_e:
        print(f"Error during stitching chunk {i}: {stitch_e}")
        wpd_successful = False


    # Update time offset for the next chunk
    if i < len(upsampled_chunks) - 1:
        current_recon_time_start += time_advance_per_chunk

print("\nWavelet reconstruction and stitching loop complete.")
if not wpd_successful:
    print("Warning: Issues encountered during WPD or stitching.")


# --- Plot Sum of Windows ---
plt.figure(figsize=(12, 4))
time_axis_sumwin = np.arange(len(sum_of_windows)) / fs_recon * 1e6
plt.plot(time_axis_sumwin, sum_of_windows, label='Sum of Windows')
plt.title('Sum of Windows Across Reconstructed Signal'); plt.xlabel('Time (µs)')
plt.ylabel('Window Sum Magnitude'); plt.grid(True); plt.show()


# --- Normalization ---
print("\n--- Normalizing reconstructed signal ---")
print("Signal Stats BEFORE Normalization:")
rms_before_norm = np.nan
max_abs_before_norm = np.nan
if np.all(np.isfinite(reconstructed_signal)):
    rms_before_norm = np.sqrt(np.mean(np.real(reconstructed_signal * np.conj(reconstructed_signal))))
    max_abs_before_norm = np.max(np.abs(reconstructed_signal)) if len(reconstructed_signal) > 0 else 0
    print(f"  RMS: {rms_before_norm:.4e}")
    print(f"  Max Abs: {max_abs_before_norm:.4e}")
else: print(f"  Contains {np.sum(~np.isfinite(reconstructed_signal))} non-finite values!")

# Create divisor, handle near-zero sums
sum_of_windows_divisor = sum_of_windows.copy()
# Find reliable regions to estimate a fallback divisor (median is robust)
reliable_threshold = 1e-4
reliable_indices = np.where(sum_of_windows_divisor >= reliable_threshold)[0]
fallback_divisor = 1.0
if len(reliable_indices) > 0:
     median_reliable_sum = np.median(sum_of_windows_divisor[reliable_indices])
     if np.isfinite(median_reliable_sum) and median_reliable_sum > 1e-9:
         fallback_divisor = median_reliable_sum
         print(f"Using median sum_of_windows ({fallback_divisor:.4f}) as fallback divisor.")
     else: print(f"Warning: Median reliable sum ({median_reliable_sum}) invalid. Using fallback {fallback_divisor}.")
else: print(f"Warning: No reliable regions found. Using fallback {fallback_divisor}.")

# Apply fallback where sum is low or non-finite
unreliable_indices = np.where((sum_of_windows_divisor < reliable_threshold) | (~np.isfinite(sum_of_windows_divisor)))[0]
sum_of_windows_divisor[unreliable_indices] = fallback_divisor
# Ensure no exact zeros remain
zero_indices = np.where(np.abs(sum_of_windows_divisor) < 1e-15)[0]
sum_of_windows_divisor[zero_indices] = fallback_divisor

# Perform normalization safely
reconstructed_signal_normalized = np.zeros_like(reconstructed_signal)
valid_divisor_indices = np.abs(sum_of_windows_divisor) > 1e-15
np.divide(reconstructed_signal, sum_of_windows_divisor, out=reconstructed_signal_normalized, where=valid_divisor_indices)

# Check result
normalization_successful = True
if not np.all(np.isfinite(reconstructed_signal_normalized)):
     num_non_finite_after = np.sum(~np.isfinite(reconstructed_signal_normalized))
     print(f"*** WARNING: {num_non_finite_after} non-finite values AFTER normalization! Zeroing them. ***")
     reconstructed_signal_normalized[~np.isfinite(reconstructed_signal_normalized)] = 0
     normalization_successful = False

print("\nSignal Stats AFTER Normalization:")
rms_after_norm = np.nan
max_abs_after_norm = np.nan
if np.all(np.isfinite(reconstructed_signal_normalized)):
    rms_after_norm = np.sqrt(np.mean(np.real(reconstructed_signal_normalized * np.conj(reconstructed_signal_normalized))))
    max_abs_after_norm = np.max(np.abs(reconstructed_signal_normalized)) if len(reconstructed_signal_normalized)>0 else 0
    print(f"  RMS: {rms_after_norm:.4e}")
    print(f"  Max Abs: {max_abs_after_norm:.4e}")
    # Compare to initial target RMS
    if EXPECTED_RMS_AFTER_SCALING > 1e-12:
         print(f"  Ratio to Initial Target RMS ({EXPECTED_RMS_AFTER_SCALING:.4e}): {rms_after_norm/EXPECTED_RMS_AFTER_SCALING:.4f}")
else: print("  Contains non-finite values after zeroing!")

reconstructed_signal = reconstructed_signal_normalized.copy() # Update main signal variable
print("\nNormalization complete.")


# --- 6. Evaluation & Visualization ---
print("\n--- Evaluating Reconstruction ---")

# Regenerate Ground Truth (using the same logic as previous script)
print("Regenerating ground truth baseband for comparison...")
gt_duration = total_duration_recon_est # Use the estimated duration
num_samples_gt_compare = int(round(gt_duration * fs_recon))
t_gt_compare = np.linspace(0, gt_duration, num_samples_gt_compare, endpoint=False)
gt_baseband_compare = np.zeros(num_samples_gt_compare, dtype=np.complex128)
mod = global_attrs.get('modulation', 'qam16')
bw_gt = global_attrs.get('total_signal_bandwidth_hz', None)
gt_target_rms = EXPECTED_RMS_AFTER_SCALING # Target RMS for GT should match initial scaling

if bw_gt is None:
    print("Error: Ground truth bandwidth missing. Cannot generate GT.")
    # Use noise as fallback GT
    gt_baseband_compare = (np.random.randn(num_samples_gt_compare) + 1j*np.random.randn(num_samples_gt_compare)) * gt_target_rms / np.sqrt(2)
else:
    if mod.lower() == 'qam16':
        symbol_rate_gt = bw_gt # Assume symbol rate = BW (adjust if necessary)
        print(f"Using GT Symbol Rate = {symbol_rate_gt/1e6:.2f} Msps")
        num_symbols_gt = int(np.ceil(gt_duration * symbol_rate_gt))
        if num_symbols_gt > 0:
            qam_points = [-3, -1, 1, 3]
            symbols = (np.random.choice(qam_points, num_symbols_gt) + 1j*np.random.choice(qam_points, num_symbols_gt)) / np.sqrt(10)
            samples_per_symbol_gt = fs_recon / symbol_rate_gt
            indices = np.floor(np.arange(num_samples_gt_compare) / samples_per_symbol_gt).astype(int)
            indices = np.minimum(indices, num_symbols_gt - 1); indices = np.maximum(indices, 0)
            gt_baseband_compare = symbols[indices] * gt_target_rms # Scale to target RMS
            rms_gt = np.sqrt(np.mean(np.real(gt_baseband_compare*np.conj(gt_baseband_compare))))
            print(f"Scaled GT baseband. Target RMS: {gt_target_rms:.4e}, Actual RMS: {rms_gt:.4e}")
        else: print("Error: Zero GT symbols calculated.")
    else: print(f"Warning: GT regeneration not implemented for {mod}. Using noise."); gt_baseband_compare = (np.random.randn(num_samples_gt_compare) + 1j*np.random.randn(num_samples_gt_compare)) * gt_target_rms / np.sqrt(2)


# Calculate Metrics (using reliable regions if possible, else full signal)
mse = np.inf; nmse = np.inf; evm_percent = np.inf
recon_eval = reconstructed_signal # Use the final normalized signal
gt_eval = gt_baseband_compare

# Ensure lengths match for comparison
min_len_eval = min(len(recon_eval), len(gt_eval))
recon_eval = recon_eval[:min_len_eval]
gt_eval = gt_eval[:min_len_eval]

# Use reliable indices if available and valid
eval_indices = slice(None) # Default to all samples
if 'reliable_indices' in locals() and len(reliable_indices)>1:
     max_reliable_idx = min(np.max(reliable_indices), min_len_eval - 1)
     valid_reliable_indices = reliable_indices[reliable_indices <= max_reliable_idx]
     if len(valid_reliable_indices) > 1:
         eval_indices = valid_reliable_indices
         print(f"Evaluating metrics using {len(valid_reliable_indices)} reliable samples.")
     else: print("Evaluating metrics using all samples (reliable indices invalid).")
else: print("Evaluating metrics using all samples.")

gt_reliable = gt_eval[eval_indices]
recon_reliable = recon_eval[eval_indices]

# Check finiteness before calculating metrics
if np.all(np.isfinite(gt_reliable)) and np.all(np.isfinite(recon_reliable)) and len(gt_reliable) > 0:
    error_reliable = gt_reliable - recon_reliable
    mse = np.mean(np.real(error_reliable * np.conj(error_reliable)))
    mean_power_gt_reliable = np.mean(np.real(gt_reliable * np.conj(gt_reliable)))
    if mean_power_gt_reliable > 1e-20:
        nmse = mse / mean_power_gt_reliable
        if nmse >= 0: evm_percent = np.sqrt(nmse) * 100
else: print("Warning: Cannot calculate metrics due to non-finite values or zero length in evaluation segments.")

print(f"\nEvaluation Metrics:")
print(f"  MSE : {mse:.4e}")
if np.isfinite(nmse): print(f"  NMSE: {nmse:.4e} ({10*np.log10(nmse):.2f} dB)")
else: print(f"  NMSE: Infinite / Undefined")
print(f"  EVM : {evm_percent:.2f}%")


# Align reconstructed signal amplitude FOR PLOTTING ONLY
reconstructed_signal_aligned_plot = reconstructed_signal.copy()
plot_align_factor = 1.0
if np.all(np.isfinite(recon_reliable)) and len(recon_reliable) > 0:
    mean_power_recon_reliable = np.mean(np.real(recon_reliable * np.conj(recon_reliable)))
    if mean_power_recon_reliable > 1e-20 and mean_power_gt_reliable > 1e-20:
        plot_align_factor = np.sqrt(mean_power_gt_reliable / mean_power_recon_reliable)
        reconstructed_signal_aligned_plot *= plot_align_factor
        print(f"Applied plotting alignment scale factor: {plot_align_factor:.4f}")
    else: print("Warning: Cannot align powers for plotting due to zero power.")
else: print("Warning: Cannot align powers for plotting due to non-finite/empty reliable segment.")

# --- Plotting (Matplotlib) ---
print("\n--- Generating Matplotlib Plots ---")
plt.style.use('seaborn-v0_8-darkgrid')
fig_mpl, axs_mpl = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
plot_samples = min(plot_length, len(t_gt_compare), len(reconstructed_signal_aligned_plot))
if plot_samples > 0:
    time_axis_plot = t_gt_compare[:plot_samples] * 1e6
    gt_plot_data = gt_eval[:plot_samples] # Use eval data which has matched length
    recon_plot_data = reconstructed_signal_aligned_plot[:plot_samples]

    # Safety replace non-finite for plotting
    gt_plot_data_safe = np.nan_to_num(gt_plot_data)
    recon_plot_data_safe = np.nan_to_num(recon_plot_data)

    axs_mpl[0].plot(time_axis_plot, np.real(gt_plot_data_safe), label='GT (Real)')
    axs_mpl[0].plot(time_axis_plot, np.imag(gt_plot_data_safe), label='GT (Imag)', alpha=0.7)
    axs_mpl[0].set_title(f'Ground Truth (First {plot_samples} samples)'); axs_mpl[0].set_ylabel('Amplitude'); axs_mpl[0].legend(fontsize='small'); axs_mpl[0].grid(True)
    max_abs_gt_plot = np.max(np.abs(gt_plot_data_safe)) if len(gt_plot_data_safe)>0 else 1.0
    ylim_gt = max(max_abs_gt_plot * 1.2, 0.05); axs_mpl[0].set_ylim(-ylim_gt, ylim_gt)

    axs_mpl[1].plot(time_axis_plot, np.real(recon_plot_data_safe), label='Recon (Real)')
    axs_mpl[1].plot(time_axis_plot, np.imag(recon_plot_data_safe), label='Recon (Imag)', alpha=0.7)
    axs_mpl[1].set_title(f'Reconstructed (Aligned for Plot) - EVM: {evm_percent:.2f}%'); axs_mpl[1].set_ylabel('Amplitude'); axs_mpl[1].legend(fontsize='small'); axs_mpl[1].grid(True)
    axs_mpl[1].set_ylim(axs_mpl[0].get_ylim()) # Match GT Y limits

    error_signal = gt_plot_data_safe - recon_plot_data_safe
    error_signal_safe = np.nan_to_num(error_signal)
    axs_mpl[2].plot(time_axis_plot, np.real(error_signal_safe), label='Error (Real)')
    axs_mpl[2].plot(time_axis_plot, np.imag(error_signal_safe), label='Error (Imag)', alpha=0.7)
    axs_mpl[2].set_title('Error (GT - Recon Aligned)'); axs_mpl[2].set_xlabel('Time (µs)'); axs_mpl[2].set_ylabel('Amplitude'); axs_mpl[2].legend(fontsize='small'); axs_mpl[2].grid(True)
    max_abs_err_plot = np.max(np.abs(error_signal_safe)) if len(error_signal_safe)>0 else 1.0
    ylim_err = max(max_abs_err_plot * 1.2, 0.02); axs_mpl[2].set_ylim(-ylim_err, ylim_err)

    plt.tight_layout(); plt.show()
else: print("Skipping Matplotlib time plot: Not enough samples.")

# --- Plot Spectra ---
plt.figure(figsize=(12, 7))
plot_spectrum_flag = False
# Recon Spectrum (Aligned for Plot)
if len(reconstructed_signal_aligned_plot) > 1:
    f_recon, spec_recon_db = compute_spectrum(reconstructed_signal_aligned_plot, fs_recon)
    if len(f_recon) > 0: plt.plot(f_recon/1e6, spec_recon_db, label='Recon Spec (Aligned)', ls='--', alpha=0.8); plot_spectrum_flag = True
else: print("Skipping reconstructed spectrum plot (aligned): Insufficient data.")
# GT Spectrum
if len(gt_eval) > 1:
    f_gt, spec_gt_db = compute_spectrum(gt_eval, fs_recon)
    if len(f_gt) > 0: plt.plot(f_gt/1e6, spec_gt_db, label='GT Spec', alpha=0.8, color='C0'); plot_spectrum_flag = True
else: print("Skipping GT spectrum plot: Insufficient data.")

if plot_spectrum_flag:
    plt.title('Spectra Comparison'); plt.xlabel('Frequency (MHz)'); plt.ylabel('Magnitude (dB)')
    plt.ylim(spectrum_ylim, 5); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
else: print("No data for spectrum plot.")


# --- Plotly Time Domain Plot ---
print("\n--- Generating Plotly Time Domain Plot for Comparison ---")
if plot_samples > 0:
    fig_plotly = go.Figure()
    fig_plotly.add_trace(go.Scatter(x=time_axis_plot, y=np.real(gt_plot_data_safe), mode='lines', name='GT (Real)', line=dict(color='blue')))
    fig_plotly.add_trace(go.Scatter(x=time_axis_plot, y=np.imag(gt_plot_data_safe), mode='lines', name='GT (Imag)', line=dict(color='lightblue', dash='dash')))
    fig_plotly.add_trace(go.Scatter(x=time_axis_plot, y=np.real(recon_plot_data_safe), mode='lines', name='Recon (Real)', line=dict(color='red')))
    fig_plotly.add_trace(go.Scatter(x=time_axis_plot, y=np.imag(recon_plot_data_safe), mode='lines', name='Recon (Imag)', line=dict(color='orange', dash='dash')))
    fig_plotly.update_layout(title=f'Plotly Time Domain Comparison (First {plot_samples} samples)', xaxis_title='Time (µs)', yaxis_title='Amplitude', legend_title='Signal', hovermode='x unified')
    try:
        mpl_ylim_recon = axs_mpl[1].get_ylim()
        print(f"Reference: Matplotlib Recon Y-Axis Limits: ({mpl_ylim_recon[0]:.4f}, {mpl_ylim_recon[1]:.4f})")
    except Exception: pass # Ignore if Matplotlib plot failed
    fig_plotly.show()
else: print("Skipping Plotly plot: Not enough samples.")


print("\nScript finished.")