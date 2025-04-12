"""
Key Changes Summary:
Pilot Tone Parameters: Added add_pilot_tone, pilot_offset_hz, pilot_power_db at the top.
Add Pilot to Ground Truth: After generating the main baseband_signal_main, it calculates the pilot amplitude and frequency (relative to baseband center) and adds the complex exponential pilot_signal to it before the final normalization.
Normalization: The combined signal (main + pilot) is normalized to RMS=1. This is important so that the specified snr_db is correctly applied relative to the total signal power present before noise addition.
Phase Offset Simulation: The phase offset logic now explicitly uses total_phase_offset = 0.0 when CONFIG_CHOICE == 'perfect_clock', simulating the ideal reference lock. (The random jump logic remains if you switch CONFIG_CHOICE).
Metadata/Filename: Updated to store pilot parameters and include _PILOT in the filename.
Plotting: Added vertical lines (axvline) to indicate the absolute RF frequency of the pilot tone on the spectrum plot for visual reference. Legend handling is slightly improved.

"""

# PC_Simulated_data_generator_clocked.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import h5py
from tqdm import tqdm
import sys
import os

print("--- Running Modified Data Generator (Simulating Perfect Clock + PILOT TONE) ---")

# --- Simulation Parameters ---
f_rf_center = 25e9
bw_total_signal = 400e6
modulation = 'qam16'
sdr_ibw = 56e6
adc_bits = 12
adc_vref = 1.0
num_chunks = 8
overlap_factor = 0.1
snr_db = 25 # SNR relative to main signal power
duration_per_chunk = 100e-6
tuning_delay = 5e-6

# --- VVVVV PILOT TONE PARAMETERS VVVVV ---
add_pilot_tone = True
# Offset from the *center* of the total signal bandwidth
pilot_offset_hz = 10e6 # Place it 80% towards lower edge
# Power relative to the main signal's normalized power (RMS=1)
pilot_power_db = -5 # dB relative to main signal power
# --- ^^^^^ PILOT TONE PARAMETERS ^^^^^ ---

# Simulation type (controls phase jumps between chunks)
CONFIG_CHOICE = 'perfect_clock' # Keep perfect clock for this test
OUTPUT_SUFFIX = '_PERFECT_CLOCK_PILOT' if CONFIG_CHOICE == 'perfect_clock' else '_RANDOM_PHASE_PILOT'
phase_offset_std_dev_rad_simulated = 0.0 # No random jumps for perfect clock

# --- Derived Parameters (Recalculated) ---
chunk_spacing = sdr_ibw * (1.0 - overlap_factor)
actual_num_chunks_needed = int(np.ceil((bw_total_signal - sdr_ibw) / chunk_spacing)) + 1
if actual_num_chunks_needed > num_chunks:
    print(f"Warning: Using {num_chunks} chunks, may not cover full BW (need {actual_num_chunks_needed}).")

total_covered_bw = chunk_spacing * (num_chunks - 1) + sdr_ibw
start_rf_chunk_center = f_rf_center - (chunk_spacing * (num_chunks - 1)) / 2
rf_chunk_centers = np.linspace(start_rf_chunk_center,
                               start_rf_chunk_center + chunk_spacing * (num_chunks - 1),
                               num_chunks)
f_lo = np.floor((np.min(rf_chunk_centers) - sdr_ibw * 0.6) / 1e9) * 1e9
if_chunk_centers = rf_chunk_centers - f_lo
if np.any(if_chunk_centers <= 0): raise ValueError("Cannot achieve positive IF.")

oversample_factor_sdr = 1.5
fs_sdr = sdr_ibw * (2 * oversample_factor_sdr)
num_samples_per_chunk = int(round(duration_per_chunk * fs_sdr))

max_if_edge_freq = np.max(if_chunk_centers) + sdr_ibw / 2
fs_ground_truth = max(2 * max_if_edge_freq * 1.5, bw_total_signal * 2 * 1.1)
total_sim_time_needed = (duration_per_chunk * num_chunks) + (tuning_delay * max(0, num_chunks - 1))
num_samples_ground_truth = int(np.ceil(total_sim_time_needed * fs_ground_truth))
t_ground_truth = np.linspace(0, total_sim_time_needed, num_samples_ground_truth, endpoint=False)

decimation_factor_float = fs_ground_truth / fs_sdr
decimation_factor = int(round(decimation_factor_float))
if not np.isclose(decimation_factor_float, decimation_factor): print(f"Warning: Non-integer decimation factor {decimation_factor_float:.4f}. Using {decimation_factor}.")

# Filter params
bpf_order = 4; bpf_stop_atten_db = 40
decim_filter_type = 'fir'; decim_zero_phase = False
decimation_filter_len_approx = 30 * decimation_factor
bpf_filter_len_approx = bpf_order * 10
filter_buffer_samples = decimation_filter_len_approx + bpf_filter_len_approx
min_len_for_processing = filter_buffer_samples + decimation_factor
estimated_needed_gt_samples_per_slice = int(round(duration_per_chunk * fs_ground_truth)) + filter_buffer_samples

print(f"\n--- Parameters Summary ---")
print(f"Simulation Type: {CONFIG_CHOICE}")
print(f"Pilot Tone Added: {add_pilot_tone} (Offset: {pilot_offset_hz/1e6:.2f} MHz, Power: {pilot_power_db} dB)")
# ... (rest of summary prints) ...
print(f"Ground Truth Sample Rate: {fs_ground_truth / 1e6:.2f} MHz")
print(f"Ground Truth Samples: {num_samples_ground_truth}")
print(f"Decimation Factor: {decimation_factor} (Ratio: {decimation_factor_float:.4f})")
print(f"--------------------------\n")


# --- 1. Generate Wideband Ground Truth Baseband Signal ---
print("Generating ground truth wideband baseband signal...")
baseband_signal_main = np.zeros(num_samples_ground_truth, dtype=np.complex128)

if modulation.lower() == 'qam16':
    # ... (Generate QAM16 baseband_symbols as before) ...
    symbol_rate_gt = bw_total_signal / 4
    num_symbols_gt = int(np.ceil(total_sim_time_needed * symbol_rate_gt))
    print(f"  Modulation: {modulation}, Symbol Rate: {symbol_rate_gt/1e6:.2f} Msps, Num Symbols: {num_symbols_gt}")
    symbols_real = np.random.choice([-3, -1, 1, 3], size=num_symbols_gt)
    symbols_imag = np.random.choice([-3, -1, 1, 3], size=num_symbols_gt)
    symbols = (symbols_real + 1j * symbols_imag) / np.sqrt(10)
    samples_per_symbol_gt = fs_ground_truth / symbol_rate_gt
    if samples_per_symbol_gt < 1: samples_per_symbol_gt = 1
    else: samples_per_symbol_gt = int(round(samples_per_symbol_gt))
    baseband_symbols = np.repeat(symbols, samples_per_symbol_gt)
    current_len = len(baseband_symbols)
    if current_len >= num_samples_ground_truth:
        baseband_signal_main = baseband_symbols[:num_samples_ground_truth]
    else:
        baseband_signal_main[:current_len] = baseband_symbols
elif modulation.lower() == 'noise':
    # ... (Generate bandlimited noise as before) ...
     print(f"  Modulation: Bandlimited noise")
     noise_gt = np.random.randn(num_samples_ground_truth) + 1j * np.random.randn(num_samples_ground_truth)
     nyquist_gt = fs_ground_truth / 2.0
     normalized_bw_cutoff = (bw_total_signal / 2.0) / nyquist_gt
     if normalized_bw_cutoff >= 1.0: normalized_bw_cutoff = 0.9999
     b_bw, a_bw = sig.butter(8, normalized_bw_cutoff, btype='low', analog=False)
     baseband_signal_main = sig.lfilter(b_bw, a_bw, noise_gt)
else:
    raise ValueError(f"Unsupported modulation: {modulation}")

# --- Add Pilot Tone ---
if add_pilot_tone:
    pilot_amplitude = np.sqrt(10**(pilot_power_db / 10.0)) # Amplitude relative to RMS=1 signal
    # Pilot frequency is relative to baseband center (0 Hz)
    f_pilot_bb = pilot_offset_hz
    print(f"  Adding pilot tone: Freq = {f_pilot_bb/1e6:.3f} MHz (relative to baseband center), Amplitude = {pilot_amplitude:.4f}")
    pilot_signal = pilot_amplitude * np.exp(1j * 2 * np.pi * f_pilot_bb * t_ground_truth)
    baseband_signal_combined = baseband_signal_main + pilot_signal
else:
    print("  Skipping pilot tone addition.")
    baseband_signal_combined = baseband_signal_main

# --- Normalize FINAL ground truth baseband (Signal + Pilot) ---
gt_rms_before_norm = np.sqrt(np.mean(np.abs(baseband_signal_combined)**2))
if gt_rms_before_norm > 1e-15:
    baseband_signal_complex_ground_truth = baseband_signal_combined / gt_rms_before_norm
    print(f"Ground truth baseband (Signal+Pilot) normalized to RMS=1.0 (Original Combined RMS={gt_rms_before_norm:.4e})")
else:
    print("Warning: Combined ground truth signal power is zero. Cannot normalize.")
    baseband_signal_complex_ground_truth = baseband_signal_combined

print(f"Ground truth baseband signal generated, {len(baseband_signal_complex_ground_truth)} samples.")


# --- Helper function for stats (no changes needed) ---
def print_stats(label, signal):
    # ... (same as before) ...
    if signal is not None and len(signal) > 0:
        finite_mask = np.isfinite(signal)
        if np.any(finite_mask):
            valid_signal = signal[finite_mask]
            rms = np.sqrt(np.mean(np.abs(valid_signal)**2))
            mabs = np.max(np.abs(valid_signal)) if len(valid_signal)>0 else 0
            ratio = mabs / rms if rms > 1e-12 else np.inf
            print(f"{label:<19}: RMS={rms:.4e}, MaxAbs={mabs:.4e}, Ratio={ratio:.4f} ({len(valid_signal)} finite samples)")
        else: print(f"{label:<19}: All non-finite data (len={len(signal)})")
    else: print(f"{label:<19}: Invalid data (None or len=0)")


# --- 2. Simulate RF Upconversion (Implicit) ---

# --- 3. Simulate Sequential Chunk Capture ---
print(f"\nSimulating capture of {num_chunks} chunks...")
captured_chunks_data = []
chunk_metadata = []
current_time_offset = 0.0

for i in tqdm(range(num_chunks)):
    # ... (Calculate time slice, slice GT signal - same as before) ...
    current_rf_center = rf_chunk_centers[i]
    current_if_center = if_chunk_centers[i]
    start_sample_gt = int(round(current_time_offset * fs_ground_truth))
    actual_end_sample_gt = min(start_sample_gt + estimated_needed_gt_samples_per_slice, num_samples_ground_truth)
    current_slice_len = actual_end_sample_gt - start_sample_gt
    if current_slice_len < min_len_for_processing:
        print(f"\nWarning: Slice for chunk {i} too short ({current_slice_len} < {min_len_for_processing}). Skipping.")
        current_time_offset += duration_per_chunk + tuning_delay
        continue
    time_slice_gt = t_ground_truth[start_sample_gt:actual_end_sample_gt]
    baseband_slice_gt = baseband_signal_complex_ground_truth[start_sample_gt:actual_end_sample_gt]

    # --- Downconversion, BPF, Decimate (same as before) ---
    if_signal_chunk_gt_rate = baseband_slice_gt * np.exp(1j * 2 * np.pi * current_if_center * time_slice_gt)
    nyquist_gt = fs_ground_truth / 2.0
    wp_low = (current_if_center - sdr_ibw / 2) / nyquist_gt
    wp_high = (current_if_center + sdr_ibw / 2) / nyquist_gt
    ws_low = (current_if_center - sdr_ibw / 2 * 1.5) / nyquist_gt
    ws_high = (current_if_center + sdr_ibw / 2 * 1.5) / nyquist_gt
    wp_low = max(1e-6, wp_low); ws_low = max(1e-6, ws_low)
    wp_high = min(1.0 - 1e-6, wp_high); ws_high = min(1.0 - 1e-6, ws_high)
    if wp_low >= wp_high or ws_low >= ws_high or ws_low <= 0 or ws_high >= 1:
         print(f"Warning: Chunk {i} BPF freqs invalid. Skipping BPF.")
         if_signal_chunk_filtered_gt_rate = if_signal_chunk_gt_rate
    else:
         sos_bpf = sig.cheby2(bpf_order, bpf_stop_atten_db, [wp_low, wp_high], btype='bandpass', analog=False, output='sos')
         if_signal_chunk_filtered_gt_rate = sig.sosfilt(sos_bpf, if_signal_chunk_gt_rate)
    try:
        if_signal_chunk_sdr_rate = sig.decimate(if_signal_chunk_filtered_gt_rate, decimation_factor, ftype=decim_filter_type, zero_phase=decim_zero_phase)
    except ValueError as e:
        print(f"\nError during decimate for chunk {i}: {e}. Skipping.")
        current_time_offset += duration_per_chunk + tuning_delay; continue

    # --- Trim/Pad (same as before) ---
    current_num_samples = len(if_signal_chunk_sdr_rate)
    if current_num_samples >= num_samples_per_chunk:
        start_trim_idx = current_num_samples - num_samples_per_chunk
        final_chunk_signal_before_noise = if_signal_chunk_sdr_rate[start_trim_idx:]
    else:
        pad_needed = num_samples_per_chunk - current_num_samples
        final_chunk_signal_before_noise = np.pad(if_signal_chunk_sdr_rate, (pad_needed, 0), mode='constant')

    # --- Add Noise (same as before) ---
    signal_power_chunk = np.mean(np.abs(final_chunk_signal_before_noise) ** 2)
    if signal_power_chunk < 1e-30: noise_chunk = np.zeros(num_samples_per_chunk, dtype=complex)
    else:
         snr_linear_chunk = 10 ** (snr_db / 10.0)
         noise_power_chunk = signal_power_chunk / snr_linear_chunk
         noise_std_dev_chunk = np.sqrt(noise_power_chunk / 2.0)
         noise_chunk = noise_std_dev_chunk * (np.random.randn(num_samples_per_chunk) + 1j * np.random.randn(num_samples_per_chunk))
    if_signal_noisy_chunk = final_chunk_signal_before_noise + noise_chunk

    # --- Apply Phase Offset (ZERO relative offset for perfect clock) ---
    total_phase_offset = 0.0 # No relative jump simulated
    if_signal_noisy_phase_applied = if_signal_noisy_chunk * np.exp(1j * total_phase_offset)

 # --- Simulate ADC Quantization ---
    max_abs_val_chunk = np.max([np.max(np.abs(if_signal_noisy_phase_applied.real)), np.max(np.abs(if_signal_noisy_phase_applied.imag))])
    if max_abs_val_chunk < 1e-15: max_abs_val_chunk = 1e-15
    scale_factor_chunk = (adc_vref / 2.0) / (max_abs_val_chunk * 1.1)
    signal_scaled_chunk = if_signal_noisy_phase_applied * scale_factor_chunk
    num_levels = 2 ** adc_bits
    quant_step = adc_vref / num_levels
    quantized_real_chunk = np.round(signal_scaled_chunk.real / quant_step) * quant_step
    quantized_imag_chunk = np.round(signal_scaled_chunk.imag / quant_step) * quant_step
    quantized_real_chunk = np.clip(quantized_real_chunk, -adc_vref / 2 + quant_step/2, adc_vref / 2 - quant_step/2)
    quantized_imag_chunk = np.clip(quantized_imag_chunk, -adc_vref / 2 + quant_step/2, adc_vref / 2 - quant_step/2)

    # Combine real and imag parts into the correct variable name
    final_chunk_data_scaled = quantized_real_chunk + 1j * quantized_imag_chunk


    # Rescale back to original signal level range
    final_chunk_data = final_chunk_data_scaled / scale_factor_chunk
    if i == 0:
         print_stats("After Quant/Rescale", final_chunk_data)

    # --- Store Chunk Data and Metadata ---
    captured_chunks_data.append(final_chunk_data)
    meta = {
        'chunk_index': i,
        'rf_center_freq_hz': current_rf_center,
        'if_center_freq_hz': current_if_center, # IF center freq for this chunk
        'sdr_sample_rate_hz': fs_sdr,
        'num_samples': len(final_chunk_data),
        'intended_duration_s': duration_per_chunk,
        'start_time_s': current_time_offset,
        'applied_phase_offset_rad': total_phase_offset, # Will be 0.0
        'filter_description': f"BPF: sosfilt_cheby2_ord{bpf_order}_rs{bpf_stop_atten_db}; DECIM: {decim_filter_type}_zero_phase_{decim_zero_phase}",
        'pilot_tone_added': add_pilot_tone, # Store pilot info
        'pilot_offset_hz': pilot_offset_hz if add_pilot_tone else None,
        'pilot_power_db_rel_signal': pilot_power_db if add_pilot_tone else None
    }
    chunk_metadata.append(meta)

    # --- Update Time Offset ---
    current_time_offset += duration_per_chunk + tuning_delay
    # --- End Chunk Loop ---

print("\nChunk capture simulation complete.")

# --- 4. Data Storage (Using HDF5) ---
# Use environment variable for output folder, default to 'simulated_data'
output_folder = os.environ.get('SIMULATED_DATA_DIR', 'simulated_data')
os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists
output_filename = f"{output_folder}/simulated_chunks_{f_rf_center / 1e9:.0f}GHz_{bw_total_signal / 1e6:.0f}MHzBW_{modulation}_sdr{sdr_ibw / 1e6:.0f}MHz.h5"
print(f"Saving chunk data and metadata to: {output_filename}")
try:
    with h5py.File(output_filename, 'w') as f:
        # Store global attributes
        # ... (copy essential global attributes as before) ...
        f.attrs['rf_center_freq_hz'] = f_rf_center
        f.attrs['total_signal_bandwidth_hz'] = bw_total_signal
        f.attrs['modulation'] = modulation
        f.attrs['sdr_ibw_hz'] = sdr_ibw
        f.attrs['sdr_sample_rate_hz'] = fs_sdr
        f.attrs['num_chunks'] = num_chunks
        f.attrs['overlap_factor'] = overlap_factor
        f.attrs['snr_db_per_chunk'] = snr_db
        f.attrs['adc_bits'] = adc_bits
        f.attrs['tuning_delay_s'] = tuning_delay
        f.attrs['phase_offset_std_dev_rad_simulated'] = phase_offset_std_dev_rad_simulated # 0.0
        # f.attrs['phase_noise_channel_std_dev_rad'] = phase_noise_channel_std_dev_rad # Info only
        f.attrs['ground_truth_sample_rate_hz'] = fs_ground_truth
        f.attrs['num_ground_truth_samples'] = num_samples_ground_truth
        f.attrs['actual_num_chunks_saved'] = len(captured_chunks_data)
        f.attrs['filter_description'] = f"BPF: sosfilt_cheby2_ord{bpf_order}_rs{bpf_stop_atten_db}; DECIM: {decim_filter_type}_zero_phase_{decim_zero_phase}"
        f.attrs['simulation_type'] = CONFIG_CHOICE # 'perfect_clock' or 'random_phase_jump'
        f.attrs['pilot_tone_added'] = add_pilot_tone
        f.attrs['pilot_offset_hz'] = pilot_offset_hz if add_pilot_tone else None
        f.attrs['pilot_power_db_rel_signal'] = pilot_power_db if add_pilot_tone else None
        f.attrs['fixed_lo_freq_hz'] = f_lo # Store the LO used


        # Store each chunk and its metadata
        for i_chunk, chunk_data in enumerate(captured_chunks_data):
            group_name = f'chunk_{i_chunk:03d}'
            group = f.create_group(group_name)
            group.create_dataset('iq_data', data=chunk_data, compression='gzip')
            if i_chunk < len(chunk_metadata):
                 for key, value in chunk_metadata[i_chunk].items():
                     if value is not None: # Avoid storing None attributes
                        group.attrs[key] = value

    print("Data saved successfully.")
except Exception as e:
    print(f"\nError saving HDF5 file: {e}")
    sys.exit(1)


# --- 5. Visualization (Spectrum Plot - adapted for pilot) ---
plot_results = True
if plot_results and captured_chunks_data:
    print("Plotting spectrum...")
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(14, 8))
    max_overall_time_mag = 0.0 # Find max magnitude in time domain for reference
    num_plotted = 0

    for chunk_data in captured_chunks_data:
        if chunk_data is not None and len(chunk_data) > 0 and np.all(np.isfinite(chunk_data)):
            mag = np.max(np.abs(chunk_data))
            if mag > max_overall_time_mag: max_overall_time_mag = mag
            num_plotted += 1

    if num_plotted == 0: print("No valid chunks to plot.")
    if max_overall_time_mag < 1e-15: max_overall_time_mag = 1.0 # Avoid zero reference

    if num_plotted > 0:
        for i_chunk, chunk_data in enumerate(captured_chunks_data):
             if chunk_data is None or len(chunk_data) < 2 or not np.all(np.isfinite(chunk_data)): continue
             if i_chunk < len(chunk_metadata): meta = chunk_metadata[i_chunk]
             else: continue

             n_fft = len(chunk_data)
             freqs_if = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1 / meta['sdr_sample_rate_hz']))
             rf_freqs = freqs_if + meta['rf_center_freq_hz'] # Shift to conceptual RF
             spectrum = np.fft.fftshift(np.fft.fft(chunk_data))
             # Plot dB relative to the max time domain signal amplitude
             spectrum_db = 20 * np.log10(np.abs(spectrum) / (n_fft * max_overall_time_mag) + 1e-12)

             plt.plot(rf_freqs / 1e9, spectrum_db, label=f'Chunk {i_chunk} (RF: {meta["rf_center_freq_hz"] / 1e9:.3f} GHz)', alpha=0.7)

             # Indicate pilot tone location if added
             if meta.get('pilot_tone_added', False):
                  pilot_rf = meta['rf_center_freq_hz'] + meta['pilot_offset_hz'] # Calculate absolute pilot RF
                  plt.axvline(pilot_rf / 1e9, color='lime', linestyle=':', linewidth=1, alpha=0.6, label=f'_Pilot {i_chunk}' if i_chunk==0 else None)


        plt.title(f'Spectra of Captured Chunks ({CONFIG_CHOICE}, Shifted to RF)\nMod: {modulation}, SNR: {snr_db} dB, Pilot: {pilot_power_db if add_pilot_tone else "None"} dB')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Magnitude (dB rel. Peak Time Signal)')
        plt.ylim(bottom=-140)
        plt.grid(True, which='both', linestyle='--')
        # Create a clean legend
        handles, labels = plt.gca().get_legend_handles_labels()
        # Remove duplicate labels (like '_Pilot x')
        by_label = dict(zip(labels, handles))
        if len(by_label) > 8: # Avoid overly cluttered legend
             plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.04, 1), loc="upper left", fontsize='small'); plt.tight_layout(rect=[0, 0, 0.85, 1])
        else:
             plt.legend(by_label.values(), by_label.keys(), fontsize='small'); plt.tight_layout()
        plt.show()

print("\nScript finished.")