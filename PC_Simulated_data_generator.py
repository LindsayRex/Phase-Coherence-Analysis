import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import h5py  # Using HDF5 for better structured storage
from tqdm import tqdm  # Optional: for progress bar
import sys # Added for exit

# --- Simulation Parameters ---
# ... (Keep parameters the same as before) ...
f_rf_center = 25e9
bw_total_signal = 400e6
modulation = 'qam16'
sdr_ibw = 56e6
adc_bits = 12
adc_vref = 1.0
num_chunks = int(np.ceil(bw_total_signal / sdr_ibw))
overlap_factor = 0.1
chunk_spacing = sdr_ibw * (1.0 - overlap_factor)
total_covered_bw = chunk_spacing * (num_chunks - 1) + sdr_ibw
print(f"Attempting to cover {bw_total_signal / 1e6:.1f} MHz using {num_chunks} chunks.")
print(f"Chunk spacing: {chunk_spacing / 1e6:.1f} MHz (IBW={sdr_ibw / 1e6:.1f} MHz, Overlap={overlap_factor * 100:.0f}%)")
print(f"Total spectral span covered by chunks: {total_covered_bw / 1e6:.1f} MHz")
start_rf_chunk_center = f_rf_center - (chunk_spacing * (num_chunks - 1)) / 2
rf_chunk_centers = np.linspace(start_rf_chunk_center,
                               start_rf_chunk_center + chunk_spacing * (num_chunks - 1),
                               num_chunks)
f_lo = 24.0e9
if_chunk_centers = rf_chunk_centers - f_lo
print(f"RF Chunk Centers (GHz): {[f / 1e9 for f in rf_chunk_centers]}")
print(f"IF Chunk Centers (MHz): {[f / 1e6 for f in if_chunk_centers]}")
if np.any(if_chunk_centers <= sdr_ibw / 2):
    raise ValueError("IF frequencies too low or negative. Adjust f_lo.")
oversample_factor_sdr = 1.5
fs_sdr = sdr_ibw * (2 * oversample_factor_sdr)
print(f"SDR ADC Sample Rate: {fs_sdr / 1e6:.2f} MHz")
snr_db = 25
duration_per_chunk = 100e-6
num_samples_per_chunk = int(duration_per_chunk * fs_sdr)
tuning_delay = 5e-6
phase_noise_std_dev_rad = np.deg2rad(1.0)
max_if_edge_freq = np.max(if_chunk_centers) + sdr_ibw / 2
print(f"Max IF edge frequency before decimation: {max_if_edge_freq / 1e6:.2f} MHz")
fs_ground_truth_nyquist_if = 2 * max_if_edge_freq
oversample_factor_gt = 1.5
fs_ground_truth = fs_ground_truth_nyquist_if * oversample_factor_gt
print(f"Required Ground Truth Sim Sample Rate (for IF): {fs_ground_truth / 1e6:.2f} MHz")
fs_check_baseband = bw_total_signal * (2 * 1.1)
if fs_ground_truth < fs_check_baseband:
    fs_ground_truth = fs_check_baseband
total_sim_time_needed = (duration_per_chunk * num_chunks) + (tuning_delay * (num_chunks - 1))
num_samples_ground_truth = int(total_sim_time_needed * fs_ground_truth)
t_ground_truth = np.linspace(0, total_sim_time_needed, num_samples_ground_truth, endpoint=False)

print(f"\n--- Parameters ---")
print(f"Total Signal BW: {bw_total_signal / 1e6:.1f} MHz @ RF Center {f_rf_center / 1e9:.2f} GHz")
print(f"SDR IBW: {sdr_ibw / 1e6:.1f} MHz")
print(f"SDR Sample Rate: {fs_sdr / 1e6:.1f} MHz")
print(f"Number of Chunks: {num_chunks}")
print(f"Duration per Chunk: {duration_per_chunk * 1e6:.1f} µs")
print(f"Samples per Chunk: {num_samples_per_chunk}")
print(f"Tuning Delay: {tuning_delay * 1e6:.1f} µs")
print(f"Phase Noise Std Dev: {np.rad2deg(phase_noise_std_dev_rad):.1f} deg")
print(f"SNR per Chunk: {snr_db} dB (Relative to signal power *after* BPF/Decimation)") # <<< CHANGE: Clarified SNR meaning
print(f"ADC Bits: {adc_bits}")
print(f"------------------\n")

# --- 1. Generate Wideband Ground Truth Baseband Signal ---
print("Generating ground truth wideband baseband signal...")
# ... (Signal generation code remains the same) ...
generated_baseband = None
baseband_signal_complex_ground_truth = np.zeros(num_samples_ground_truth, dtype=complex)

if modulation.lower() == 'qam16':
    symbol_rate_gt = bw_total_signal / 4
    num_symbols_gt = int(np.ceil(total_sim_time_needed * symbol_rate_gt))
    symbols_real = np.random.choice([-3, -1, 1, 3], size=num_symbols_gt)
    symbols_imag = np.random.choice([-3, -1, 1, 3], size=num_symbols_gt)
    symbols = (symbols_real + 1j * symbols_imag) / np.sqrt(10)
    samples_per_symbol_gt = int(fs_ground_truth / symbol_rate_gt)
    if samples_per_symbol_gt == 0: samples_per_symbol_gt = 1
    baseband_symbols = np.repeat(symbols, samples_per_symbol_gt)
    generated_baseband = baseband_symbols
# Add other modulation types if needed...
elif modulation.lower() == 'noise':
     noise_gt = np.random.randn(num_samples_ground_truth) + 1j * np.random.randn(num_samples_ground_truth)
     nyquist_gt = fs_ground_truth / 2
     normalized_bw_cutoff = (bw_total_signal / 2) / nyquist_gt
     if normalized_bw_cutoff >= 1.0: normalized_bw_cutoff = 0.999
     b_bw, a_bw = sig.butter(5, normalized_bw_cutoff, btype='low', analog=False)
     baseband_signal_complex_ground_truth = sig.lfilter(b_bw, a_bw, noise_gt)

else: raise ValueError(f"Unsupported modulation: {modulation}")

if modulation.lower() not in ['noise']: # Handle generated baseband assignment
    if generated_baseband is None: raise RuntimeError("Baseband signal generation failed.")
    current_len = len(generated_baseband)
    if current_len > num_samples_ground_truth:
        baseband_signal_complex_ground_truth = generated_baseband[:num_samples_ground_truth]
    elif current_len < num_samples_ground_truth:
        pad_width = num_samples_ground_truth - current_len
        baseband_signal_complex_ground_truth = np.pad(generated_baseband, (0, pad_width), mode='constant', constant_values=(0 + 0j))
    else:
        baseband_signal_complex_ground_truth = generated_baseband

if len(baseband_signal_complex_ground_truth) != num_samples_ground_truth:
     raise RuntimeError(f"FATAL: Final baseband length mismatch")

gt_rms_before_norm = np.sqrt(np.mean(np.abs(baseband_signal_complex_ground_truth)**2))
if gt_rms_before_norm > 1e-20:
    baseband_signal_complex_ground_truth /= gt_rms_before_norm # Normalize RMS to 1.0 BEFORE any processing
else: print("Warning: Ground truth signal power is near zero. Skipping normalization.")

print(f"Ground truth baseband signal generated, {len(baseband_signal_complex_ground_truth)} samples.")

# --- 2. Simulate RF Upconversion (Conceptual) ---
# --- 3. Simulate Sequential Chunk Capture ---
print(f"\nSimulating capture of {num_chunks} chunks...")
captured_chunks_data = []
chunk_metadata = []
current_time_offset = 0.0
last_chunk_phase_offset = 0.0

# Decimation setup
decimation_factor_float = fs_ground_truth / fs_sdr
decimation_factor = int(round(decimation_factor_float))
if not np.isclose(decimation_factor_float, decimation_factor):
    print(f"Warning: Non-integer decimation factor. Using {decimation_factor}. Ratio: {decimation_factor_float:.4f}")

# --- Helper function for debug prints ---
def print_stats(label, signal):
    if signal is not None and len(signal) > 0 and np.all(np.isfinite(signal)):
        rms = np.sqrt(np.mean(np.abs(signal)**2))
        mabs = np.max(np.abs(signal))
        ratio = mabs / rms if rms > 1e-12 else np.inf
        print(f"{label:<19}: RMS={rms:.4e}, MaxAbs={mabs:.4e}, Ratio={ratio:.4f}")
    else:
        print(f"{label:<19}: Invalid data (len={len(signal) if signal is not None else 0} or non-finite)")


for i in tqdm(range(num_chunks)):
    current_rf_center = rf_chunk_centers[i]
    current_if_center = if_chunk_centers[i]

    # --- Determine Time Slice ---
    start_sample_gt = int(round(current_time_offset * fs_ground_truth))
    filter_order_bpf = 8 # Corresponds to Cheby2 N=4 used with lfilter
    # <<< CHANGE: Estimate filter delay for slicing - rough estimate
    # For IIR, delay varies with freq. For FIR in decimate, it's ~ (order/2) / fs_ground_truth
    # Add a generous buffer instead of precise calculation for now
    decimation_filter_length_approx = 60 * decimation_factor # Estimate length of FIR in decimate
    filter_buffer_samples = filter_order_bpf * 5 + decimation_filter_length_approx # Generous buffer
    estimated_needed_gt_samples = int(duration_per_chunk * fs_ground_truth) + filter_buffer_samples
    potential_end_sample_gt = start_sample_gt + estimated_needed_gt_samples
    actual_end_sample_gt = min(potential_end_sample_gt, num_samples_ground_truth)
    if start_sample_gt >= actual_end_sample_gt: break
    time_slice_gt = t_ground_truth[start_sample_gt:actual_end_sample_gt]
    baseband_slice_gt = baseband_signal_complex_ground_truth[start_sample_gt:actual_end_sample_gt]
    if time_slice_gt.shape != baseband_slice_gt.shape: raise RuntimeError("Shape mismatch after slicing!")
    min_len_for_processing = filter_buffer_samples # Need enough samples for filter transients + decimation
    if len(time_slice_gt) < min_len_for_processing:
        print(f"\nWarning: Slice for chunk {i} too short ({len(time_slice_gt)} needed {min_len_for_processing}). Skipping.")
        current_time_offset += duration_per_chunk + tuning_delay
        continue

    # --- Debug Print for Chunk 0 ---
    if i == 0: print(f"\n--- Debug: Chunk {i} Processing Stages (Realistic Filters) ---")
    if i == 0: print_stats("Baseband Slice", baseband_slice_gt)

    # --- Downconversion ---
    if_signal_chunk_gt_rate = baseband_slice_gt * np.exp(1j * 2 * np.pi * current_if_center * time_slice_gt)
    if i == 0: print_stats("IF Signal (GT Rate)", if_signal_chunk_gt_rate)

    # --- Band-Pass Filter (Causal) ---
    low_cutoff = current_if_center - sdr_ibw / 2 * 1.3
    high_cutoff = current_if_center + sdr_ibw / 2 * 1.3
    if low_cutoff <= 0: low_cutoff = 1e3
    nyquist_gt = fs_ground_truth / 2.0
    wn_low = low_cutoff / nyquist_gt
    wn_high = high_cutoff / nyquist_gt
    if wn_low <= 0 or wn_high >= 1.0:
        if wn_high >= 1.0 and np.isclose(wn_high, 1.0): wn_high = 0.999999
        else: raise ValueError(f"Chunk {i}: Invalid filter freqs. Wn_low={wn_low:.4f}, Wn_high={wn_high:.4f}.")
    # Use Cheby2 order 4, 40dB stopband atten (effective order approx 8 like filtfilt)
    b_bpf, a_bpf = sig.cheby2(4, 40, [wn_low, wn_high], btype='bandpass', analog=False)
    # <<< CHANGE: Use lfilter for causal filtering >>>
    if_signal_chunk_filtered_gt_rate = sig.lfilter(b_bpf, a_bpf, if_signal_chunk_gt_rate)
    # <<< CHANGE: REMOVE RMS normalization after BPF >>>
    # rms_before_bpf = np.sqrt(np.mean(np.abs(if_signal_chunk_gt_rate)**2)) # Removed
    # rms_after_bpf = np.sqrt(np.mean(np.abs(if_signal_chunk_filtered_gt_rate)**2)) # Removed
    # if rms_after_bpf > 1e-12: # Removed
    #     if_signal_chunk_filtered_gt_rate *= rms_before_bpf / rms_after_bpf # Removed
    if i == 0: print_stats("After BPF (lfilter)", if_signal_chunk_filtered_gt_rate) # Label changed


    # --- Decimate (Causal) ---
    try:
        # <<< CHANGE: Use zero_phase=False for causal FIR decimation >>>
        if_signal_chunk_sdr_rate = sig.decimate(if_signal_chunk_filtered_gt_rate, decimation_factor, ftype='fir', zero_phase=False)
    except ValueError as e:
        print(f"\nError during decimate for chunk {i}: {e}. Input length: {len(if_signal_chunk_filtered_gt_rate)}. Skipping chunk.")
        current_time_offset += duration_per_chunk + tuning_delay
        continue
    if i == 0: print_stats("After Decimate", if_signal_chunk_sdr_rate) # Label changed, raw removed

    # <<< CHANGE: REMOVE Amplitude Correction after Decimate >>>
    # rms_after_decimate_unscaled = np.sqrt(np.mean(np.abs(if_signal_chunk_sdr_rate_unscaled)**2)) # Removed
    # decimation_scale_factor = 1.0 # Removed
    # if rms_after_decimate_unscaled > 1e-12: # Removed
    #     decimation_scale_factor = 1.0 / rms_after_decimate_unscaled # Removed
    #     #if i == 0: print(f"Decim Scale Factor : {decimation_scale_factor:.4f}") # Removed
    # elif i == 0: print("Warning: Cannot calculate decimation scale factor due to zero RMS.") # Removed
    # if_signal_chunk_sdr_rate = if_signal_chunk_sdr_rate_unscaled * decimation_scale_factor # Removed
    # if i == 0: print_stats("After Decim Scale", if_signal_chunk_sdr_rate) # Removed Print

    # --- Trim/Pad (Takes first N samples, implicitly handles delay) ---
    # Need to potentially trim MORE from the start due to filter delay
    # Let's keep the simple logic for now: take the first num_samples_per_chunk
    # This assumes the desired signal portion is still within this block after delays.
    # A more precise method would estimate group delay and offset the trim.
    current_num_samples = len(if_signal_chunk_sdr_rate)
    if current_num_samples >= num_samples_per_chunk:
        # <<< CONSIDERATION: If group delay is large, the *start* might be transient.
        # However, trimming from the start complicates overlap. Keep simple for now.
        if_signal_chunk_sdr_rate = if_signal_chunk_sdr_rate[:num_samples_per_chunk]
    else:
        # Pad if too short (might happen at the very end of the simulation)
        pad_needed = num_samples_per_chunk - current_num_samples
        if_signal_chunk_sdr_rate = np.pad(if_signal_chunk_sdr_rate, (0, pad_needed), mode='constant', constant_values=(0+0j))
    if i == 0: print_stats("After Trim/Pad", if_signal_chunk_sdr_rate)

    # --- Add Noise ---
    # <<< CHANGE: SNR is now relative to the *actual* power after filtering/decimation >>>
    signal_power_chunk = np.mean(np.abs(if_signal_chunk_sdr_rate) ** 2)
    # Handle potential zero power case
    if signal_power_chunk < 1e-30: # Use a smaller threshold as power is not normalized
         noise_chunk = np.zeros(num_samples_per_chunk, dtype=complex)
         if i==0: print("Warning: Signal power near zero before AWGN.")
    else:
         snr_linear_chunk = 10 ** (snr_db / 10.0)
         noise_power_chunk = signal_power_chunk / snr_linear_chunk
         noise_std_dev_chunk = np.sqrt(noise_power_chunk / 2.0)
         noise_chunk = noise_std_dev_chunk * (np.random.randn(num_samples_per_chunk) + 1j * np.random.randn(num_samples_per_chunk))

    if_signal_noisy_chunk = if_signal_chunk_sdr_rate + noise_chunk
    if i == 0: print_stats("After AWGN", if_signal_noisy_chunk)

    # --- Add Phase Noise ---
    current_phase_noise = np.random.randn() * phase_noise_std_dev_rad
    total_phase_offset = last_chunk_phase_offset + current_phase_noise
    if_signal_noisy_chunk *= np.exp(1j * total_phase_offset)
    last_chunk_phase_offset = total_phase_offset # Accumulate phase offset
    if i == 0: print_stats("After Phase Noise", if_signal_noisy_chunk)

    # --- Simulate ADC Quantization ---
    # (Quantization logic remains the same - dynamic scaling based on max value)
    max_abs_val_chunk = np.max([np.max(np.abs(if_signal_noisy_chunk.real)), np.max(np.abs(if_signal_noisy_chunk.imag))])
    if max_abs_val_chunk < 1e-15: max_abs_val_chunk = 1e-15 # Avoid division by zero if signal is tiny
    scale_factor_chunk = (adc_vref / 2.0) / (max_abs_val_chunk * 1.1) # Add 10% headroom
    signal_scaled_chunk = if_signal_noisy_chunk * scale_factor_chunk
    num_levels = 2 ** adc_bits
    quant_step = adc_vref / num_levels
    quantized_real_chunk = np.round(signal_scaled_chunk.real / quant_step) * quant_step
    quantized_imag_chunk = np.round(signal_scaled_chunk.imag / quant_step) * quant_step
    quantized_real_chunk = np.clip(quantized_real_chunk, -adc_vref / 2, adc_vref / 2)
    quantized_imag_chunk = np.clip(quantized_imag_chunk, -adc_vref / 2, adc_vref / 2)
    final_chunk_data_scaled = quantized_real_chunk + 1j * quantized_imag_chunk
    # Rescale back so the stored data isn't directly tied to adc_vref but preserves quantization noise relative to signal
    final_chunk_data = final_chunk_data_scaled / scale_factor_chunk
    if i == 0:
         print_stats("After Quant/Rescale", final_chunk_data)
         print(f"----------------------------------------\n") # End debug prints for chunk 0

    # --- Store Chunk Data and Metadata ---
    captured_chunks_data.append(final_chunk_data)
    meta = {
        'chunk_index': i,
        'rf_center_freq_hz': current_rf_center,
        'if_center_freq_hz': current_if_center,
        'sdr_sample_rate_hz': fs_sdr,
        'num_samples': len(final_chunk_data),
        'intended_duration_s': duration_per_chunk,
        'start_time_s': current_time_offset,
        'applied_phase_offset_rad': total_phase_offset,
        # <<< ADDITION: Add info about filtering used >>>
        'filter_bpf_method': 'lfilter_cheby2_ord4_rs40',
        'filter_decim_method': 'fir_zero_phase_False'
    }
    chunk_metadata.append(meta)

    # --- Update Time Offset ---
    current_time_offset += duration_per_chunk + tuning_delay

print("\nChunk capture simulation complete.")

# --- 4. Data Storage (Using HDF5) ---
# Add filter info to filename? Optional.
output_filename = f"simulated_chunks_{f_rf_center / 1e9:.0f}GHz_{bw_total_signal / 1e6:.0f}MHzBW_{modulation}_sdr{sdr_ibw / 1e6:.0f}MHz.h5"
print(f"Saving chunk data and metadata to: {output_filename}")
try:
    with h5py.File(output_filename, 'w') as f:
        # Store global attributes
        f.attrs['rf_center_freq_hz'] = f_rf_center
        f.attrs['total_signal_bandwidth_hz'] = bw_total_signal
        f.attrs['modulation'] = modulation
        f.attrs['sdr_ibw_hz'] = sdr_ibw
        f.attrs['sdr_sample_rate_hz'] = fs_sdr
        f.attrs['num_chunks'] = num_chunks # Target number
        f.attrs['intended_num_chunks'] = num_chunks
        f.attrs['overlap_factor'] = overlap_factor
        f.attrs['snr_db_per_chunk'] = snr_db
        f.attrs['adc_bits'] = adc_bits
        f.attrs['tuning_delay_s'] = tuning_delay
        f.attrs['phase_noise_std_dev_rad'] = phase_noise_std_dev_rad
        f.attrs['ground_truth_sample_rate_hz'] = fs_ground_truth
        f.attrs['num_ground_truth_samples'] = num_samples_ground_truth
        f.attrs['actual_num_chunks_saved'] = len(captured_chunks_data)
        # <<< ADDITION: Store global filter info >>>
        f.attrs['filter_description'] = "Causal BPF (lfilter Cheby2 Ord4 rs40) and Causal Decimation (FIR zero_phase=False)"

        # Store each chunk and its metadata
        for i_chunk, chunk_data in enumerate(captured_chunks_data):
            group = f.create_group(f'chunk_{i_chunk:03d}')
            group.create_dataset('iq_data', data=chunk_data, compression='gzip')
            if i_chunk < len(chunk_metadata):
                 for key, value in chunk_metadata[i_chunk].items():
                     group.attrs[key] = value
            else: print(f"Warning: Metadata missing for chunk index {i_chunk} during save.")

    print("Data saved successfully.")
except Exception as e:
    print(f"\nError saving HDF5 file: {e}")
    sys.exit(1)


# --- 5. Visualization (Example: Spectrum) ---
# (Plotting code remains the same, but might show different spectral shapes due to causal filters)
plot_results = True
if plot_results and captured_chunks_data:
    print("Plotting spectrum...")
    # ... (Spectrum plotting code remains the same) ...
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(14, 8))
    max_overall_magnitude = -np.inf
    all_finite = True
    for i_chunk, chunk_data in enumerate(captured_chunks_data): # Renamed loop var
        if chunk_data is None or len(chunk_data) < 2: continue
        if not np.all(np.isfinite(chunk_data)):
             print(f"Warning: Chunk {i_chunk} contains non-finite values. Skipping spectrum plot.")
             all_finite = False; continue
        spectrum_chunk = np.fft.fftshift(np.fft.fft(chunk_data))
        current_max = np.max(np.abs(spectrum_chunk))
        if current_max > max_overall_magnitude: max_overall_magnitude = current_max

    if not all_finite: print("Skipping spectrum plot due to non-finite data.")
    elif max_overall_magnitude < 1e-12: max_overall_magnitude = 1 # Avoid log10(0)

    if all_finite:
         for i_chunk, chunk_data in enumerate(captured_chunks_data): # Renamed loop var
             if chunk_data is None or len(chunk_data) < 2: continue
             if i_chunk < len(chunk_metadata): meta = chunk_metadata[i_chunk]
             else: continue # Skip if no metadata

             f_axis_chunk = np.fft.fftshift(np.fft.fftfreq(len(chunk_data), d=1 / meta['sdr_sample_rate_hz']))
             f_axis_rf = f_axis_chunk + meta['rf_center_freq_hz']
             spectrum_chunk = np.fft.fftshift(np.fft.fft(chunk_data))
             spectrum_db = 20 * np.log10(np.abs(spectrum_chunk) / max_overall_magnitude + 1e-12)
             plt.plot(f_axis_rf / 1e9, spectrum_db, label=f'Chunk {i_chunk} (RF Center: {meta["rf_center_freq_hz"] / 1e9:.3f} GHz)')

         plt.title(f'Spectra of Captured Chunks (Realistic Filters, Shifted to RF)\nTotal BW={bw_total_signal / 1e6:.1f} MHz, SDR IBW={sdr_ibw / 1e6:.1f} MHz')
         plt.xlabel('Frequency (GHz)')
         plt.ylabel('Magnitude (dB rel. Max)')
         plt.ylim(bottom=-120) # Adjusted ylim potentially
         plt.grid(True)
         actual_num_chunks_plotted = len(captured_chunks_data)
         if actual_num_chunks_plotted > 5: plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize='small'); plt.tight_layout(rect=[0, 0, 0.82, 1])
         else: plt.legend(); plt.tight_layout()
         plt.show()

print("\nScript finished.")