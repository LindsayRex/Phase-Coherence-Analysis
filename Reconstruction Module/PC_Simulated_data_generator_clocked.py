# PC_Simulated_data_generator_clocked.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import h5py
from tqdm import tqdm
import sys
import os
import json
import logging

# Get logger for this module
logger = logging.getLogger(__name__)
# Configure basic logging if run standalone
if not logging.getLogger().hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


logger.info("--- Running Modified Data Generator (Simulating Perfect Clock + PILOT TONE + Saving GT) ---")

# --- Simulation Parameters ---
# ... (Keep all parameters as they were, e.g., f_rf_center, bw_total_signal, pilot_offset_hz, pilot_power_db, etc.) ...
f_rf_center = 25e9
bw_total_signal = 400e6
modulation = 'qam16'
sdr_ibw = 56e6
adc_bits = 12
adc_vref = 1.0
num_chunks = 8
overlap_factor = 0.1
snr_db = 25
duration_per_chunk = 100e-6
tuning_delay = 5e-6
add_pilot_tone = True
pilot_offset_hz = 10e6
pilot_power_db = -5
CONFIG_CHOICE = 'perfect_clock'
phase_offset_std_dev_rad_simulated = 0.0
phase_noise_channel_std_dev_rad = np.deg2rad(1.0)


# --- Derived Parameters ---
# ... (Keep derived parameter calculations as before) ...
chunk_spacing = sdr_ibw * (1.0 - overlap_factor)
actual_num_chunks_needed = int(np.ceil((bw_total_signal - sdr_ibw) / chunk_spacing)) + 1

if actual_num_chunks_needed > num_chunks: logger.warning(f"Using {num_chunks} chunks, may not cover full BW (need {actual_num_chunks_needed}).")
total_covered_bw = chunk_spacing * (num_chunks - 1) + sdr_ibw
start_rf_chunk_center = f_rf_center - (chunk_spacing * (num_chunks - 1)) / 2
rf_chunk_centers = np.linspace(start_rf_chunk_center, start_rf_chunk_center + chunk_spacing * (num_chunks - 1), num_chunks)
f_lo = np.floor((np.min(rf_chunk_centers) - sdr_ibw * 0.6) / 1e9) * 1e9
if_chunk_centers = rf_chunk_centers - f_lo

if np.any(if_chunk_centers <= 0): raise ValueError("Cannot achieve positive IF.")
oversample_factor_sdr = 1.5; fs_sdr = sdr_ibw * (2 * oversample_factor_sdr)
num_samples_per_chunk = int(round(duration_per_chunk * fs_sdr))
max_if_edge_freq = np.max(if_chunk_centers) + sdr_ibw / 2
fs_ground_truth = max(2 * max_if_edge_freq * 1.5, bw_total_signal * 2 * 1.1)
total_sim_time_needed = (duration_per_chunk * num_chunks) + (tuning_delay * max(0, num_chunks - 1))
num_samples_ground_truth = int(np.ceil(total_sim_time_needed * fs_ground_truth))
t_ground_truth = np.linspace(0, total_sim_time_needed, num_samples_ground_truth, endpoint=False)
decimation_factor_float = fs_ground_truth / fs_sdr; decimation_factor = int(round(decimation_factor_float))

if not np.isclose(decimation_factor_float, decimation_factor): logger.warning(f"Non-integer decimation factor {decimation_factor_float:.4f}. Using {decimation_factor}.")
bpf_order = 4; bpf_stop_atten_db = 40; decim_filter_type = 'fir'; decim_zero_phase = False
decimation_filter_len_approx = 30 * decimation_factor; bpf_filter_len_approx = bpf_order * 10
filter_buffer_samples = decimation_filter_len_approx + bpf_filter_len_approx
min_len_for_processing = filter_buffer_samples + decimation_factor
num_samples_gt_for_chunk_duration = int(round(duration_per_chunk * fs_ground_truth))
estimated_needed_gt_samples_per_slice = num_samples_gt_for_chunk_duration + filter_buffer_samples

# --- Print Summary using Logging ---
logger.info(f"\n--- Parameters Summary ---")
# ... (Log all parameters as before) ...
logger.info(f"Ground Truth Sample Rate: {fs_ground_truth / 1e6:.2f} MHz")
logger.info(f"Ground Truth Samples: {num_samples_ground_truth}")
logger.info(f"--------------------------\n")


# --- 1. Generate Wideband Ground Truth Baseband Signal ---
logger.info("Generating ground truth wideband baseband signal...")
# ... (Generate baseband_signal_main and add pilot tone as before) ...
baseband_signal_main = np.zeros(num_samples_ground_truth, dtype=np.complex128)
# ... (Code to populate baseband_signal_main with QAM16) ...
if modulation.lower() == 'qam16':
    symbol_rate_gt = bw_total_signal / 4
    num_symbols_gt = int(np.ceil(total_sim_time_needed * symbol_rate_gt))
    logger.debug(f"  Modulation: {modulation}, Symbol Rate: {symbol_rate_gt/1e6:.2f} Msps, Num Symbols: {num_symbols_gt}")
    symbols = (np.random.choice([-3,-1,1,3],size=num_symbols_gt) + 1j*np.random.choice([-3,-1,1,3],size=num_symbols_gt))/np.sqrt(10)
    samples_per_symbol_gt = int(round(fs_ground_truth / symbol_rate_gt)) if fs_ground_truth >= symbol_rate_gt else 1
    baseband_symbols = np.repeat(symbols, samples_per_symbol_gt)
    len_to_use = min(len(baseband_symbols), num_samples_ground_truth)
    baseband_signal_main[:len_to_use] = baseband_symbols[:len_to_use]

if add_pilot_tone:
    pilot_amplitude = np.sqrt(10**(pilot_power_db / 10.0))
    f_pilot_bb = pilot_offset_hz
    logger.info(f"  Adding pilot tone: Freq = {f_pilot_bb/1e6:.3f} MHz, Amplitude = {pilot_amplitude:.4f}")
    pilot_signal = pilot_amplitude * np.exp(1j * 2 * np.pi * f_pilot_bb * t_ground_truth)
    baseband_signal_combined = baseband_signal_main + pilot_signal
else: baseband_signal_combined = baseband_signal_main
# Normalize FINAL ground truth baseband
gt_rms_before_norm = np.sqrt(np.mean(np.abs(baseband_signal_combined)**2))
if gt_rms_before_norm > 1e-15:
    baseband_signal_complex_ground_truth = baseband_signal_combined / gt_rms_before_norm
    logger.info(f"Ground truth baseband (Signal+Pilot) normalized to RMS=1.0")
else: logger.warning("Combined ground truth signal power is zero."); baseband_signal_complex_ground_truth = baseband_signal_combined
logger.info(f"Ground truth baseband signal generated, {len(baseband_signal_complex_ground_truth)} samples.")


# --- 3. Simulate Sequential Chunk Capture ---
logger.info(f"\nSimulating capture of {num_chunks} chunks...")
captured_chunks_data = []
chunk_metadata = []
current_time_offset = 0.0

for i in tqdm(range(num_chunks)):
    # ... (Chunk processing loop remains exactly the same as in response #37) ...
    # ... (Assign centers, slice, check length, downconvert, BPF, decimate, trim/pad, add noise, phase offset=0, quantize) ...
    # ... (Append final_chunk_data to captured_chunks_data and meta to chunk_metadata) ...
    # --- Assign loop variables first ---
    current_rf_center = rf_chunk_centers[i]; current_if_center = if_chunk_centers[i]
    # Determine Time Slice
    start_sample_gt = int(round(current_time_offset * fs_ground_truth)); actual_end_sample_gt = min(start_sample_gt + estimated_needed_gt_samples_per_slice, num_samples_ground_truth)
    current_slice_len = actual_end_sample_gt - start_sample_gt
    if current_slice_len < min_len_for_processing: logger.warning(f"\nSlice Chk {i} short. Skipping."); current_time_offset += duration_per_chunk + tuning_delay; continue
    time_slice_gt = t_ground_truth[start_sample_gt:actual_end_sample_gt]; baseband_slice_gt = baseband_signal_complex_ground_truth[start_sample_gt:actual_end_sample_gt]
    # Downconversion
    if_signal_chunk_gt_rate = baseband_slice_gt * np.exp(1j * 2 * np.pi * current_if_center * time_slice_gt)
    # BPF
    nyquist_gt = fs_ground_truth / 2.0; wp_low = max(1e-6, (current_if_center - sdr_ibw / 2) / nyquist_gt); wp_high = min(1.0 - 1e-6, (current_if_center + sdr_ibw / 2) / nyquist_gt)
    ws_low = max(1e-6, (current_if_center - sdr_ibw / 2 * 1.5) / nyquist_gt); ws_high = min(1.0 - 1e-6, (current_if_center + sdr_ibw / 2 * 1.5) / nyquist_gt)
    if wp_low >= wp_high or ws_low >= ws_high or ws_low <= 0 or ws_high >= 1: if_signal_chunk_filtered_gt_rate = if_signal_chunk_gt_rate
    else: sos_bpf = sig.cheby2(bpf_order, bpf_stop_atten_db, [wp_low, wp_high], btype='bandpass', analog=False, output='sos'); if_signal_chunk_filtered_gt_rate = sig.sosfilt(sos_bpf, if_signal_chunk_gt_rate)
    # Decimate
    try: if_signal_chunk_sdr_rate = sig.decimate(if_signal_chunk_filtered_gt_rate, decimation_factor, ftype=decim_filter_type, zero_phase=decim_zero_phase)
    except ValueError as e: logger.error(f"\nDecimate Error Chk {i}: {e}. Skipping."); current_time_offset += duration_per_chunk + tuning_delay; continue
    # Trim/Pad
    current_num_samples = len(if_signal_chunk_sdr_rate)
    if current_num_samples >= num_samples_per_chunk: final_chunk_signal_before_noise = if_signal_chunk_sdr_rate[current_num_samples - num_samples_per_chunk:]
    else: final_chunk_signal_before_noise = np.pad(if_signal_chunk_sdr_rate, (num_samples_per_chunk - current_num_samples, 0), mode='constant')
    # Add Noise
    signal_power_chunk = np.mean(np.abs(final_chunk_signal_before_noise) ** 2)
    if signal_power_chunk < 1e-30: noise_chunk = np.zeros(num_samples_per_chunk, dtype=complex)
    else: snr_linear_chunk = 10**(snr_db / 10.0); noise_power_chunk = signal_power_chunk / snr_linear_chunk; noise_std_dev_chunk = np.sqrt(noise_power_chunk / 2.0); noise_chunk = noise_std_dev_chunk * (np.random.randn(num_samples_per_chunk) + 1j * np.random.randn(num_samples_per_chunk))
    if_signal_noisy_chunk = final_chunk_signal_before_noise + noise_chunk
    # Apply Phase Offset
    total_phase_offset = 0.0; if_signal_noisy_phase_applied = if_signal_noisy_chunk * np.exp(1j * total_phase_offset)
    # Quantization
    max_abs_val_chunk = np.max(np.abs(if_signal_noisy_phase_applied)) if len(if_signal_noisy_phase_applied)>0 else 1e-15; max_abs_val_chunk = max(max_abs_val_chunk, 1e-15)
    scale_factor_chunk = (adc_vref / 2.0) / (max_abs_val_chunk * 1.1)
    signal_scaled_chunk = if_signal_noisy_phase_applied * scale_factor_chunk; num_levels = 2 ** adc_bits; quant_step = adc_vref / num_levels
    quantized_real_chunk = np.round(signal_scaled_chunk.real / quant_step) * quant_step; quantized_imag_chunk = np.round(signal_scaled_chunk.imag / quant_step) * quant_step
    quantized_real_chunk = np.clip(quantized_real_chunk, -adc_vref / 2 + quant_step/2, adc_vref / 2 - quant_step/2); quantized_imag_chunk = np.clip(quantized_imag_chunk, -adc_vref / 2 + quant_step/2, adc_vref / 2 - quant_step/2)
    final_chunk_data = (quantized_real_chunk + 1j*quantized_imag_chunk) / scale_factor_chunk
    # Store Data/Meta
    captured_chunks_data.append(final_chunk_data)
    meta = { 'chunk_index': i, 'rf_center_freq_hz': current_rf_center, 'if_center_freq_hz': current_if_center, 'sdr_sample_rate_hz': fs_sdr, 'num_samples': len(final_chunk_data), 'intended_duration_s': duration_per_chunk, 'start_time_s': current_time_offset, 'applied_phase_offset_rad': total_phase_offset, 'pilot_tone_added': add_pilot_tone, 'pilot_offset_hz': pilot_offset_hz if add_pilot_tone else None, 'pilot_power_db_rel_signal': pilot_power_db if add_pilot_tone else None }
    chunk_metadata.append(meta)
    # Update Time Offset
    current_time_offset += duration_per_chunk + tuning_delay

logger.info("\nChunk capture simulation complete.")

# --- 4. Data Storage (Using HDF5) ---
script_dir = os.path.dirname(__file__) if "__file__" in locals() else "."
output_dir = os.path.abspath(os.path.join(script_dir, '..', 'simulated_data'))
os.makedirs(output_dir, exist_ok=True)
# Use standard filename
output_filename_base = f"simulated_chunks_{f_rf_center / 1e9:.0f}GHz_{bw_total_signal / 1e6:.0f}MHzBW_{modulation}_sdr{sdr_ibw / 1e6:.0f}MHz"
output_filename = os.path.join(output_dir, f"{output_filename_base}.h5")

logger.info(f"Saving chunk data and metadata to: {output_filename}")
try:
    with h5py.File(output_filename, 'w') as f:
        # Store global attributes
        logger.debug("Saving global attributes...")
        attrs_to_save = { # Using a dictionary for clarity
            'rf_center_freq_hz': f_rf_center, 'total_signal_bandwidth_hz': bw_total_signal,
            'modulation': modulation, 'sdr_ibw_hz': sdr_ibw, 'sdr_sample_rate_hz': fs_sdr,
            'num_chunks': num_chunks, 'intended_num_chunks': num_chunks,
            'overlap_factor': overlap_factor, 'snr_db_per_chunk': snr_db,
            'adc_bits': adc_bits, 'tuning_delay_s': tuning_delay,
            'simulation_type': CONFIG_CHOICE,
            'phase_offset_std_dev_rad_simulated': phase_offset_std_dev_rad_simulated,
            'phase_noise_channel_std_dev_rad': phase_noise_channel_std_dev_rad,
            'ground_truth_sample_rate_hz_sim': fs_ground_truth, # GT Generation Rate
            'num_ground_truth_samples_sim': num_samples_ground_truth,
            'actual_num_chunks_saved': len(captured_chunks_data),
            'filter_description': f"BPF(sosfilt_cheby2_ord{bpf_order}_rs{bpf_stop_atten_db}) DECIM(fir_zero_phase_{decim_zero_phase})",
            'fixed_lo_freq_hz': f_lo, 'pilot_tone_added': add_pilot_tone,
            'pilot_offset_hz': pilot_offset_hz if add_pilot_tone else None,
            'pilot_power_db_rel_signal': pilot_power_db if add_pilot_tone else None
        }
        for key, value in attrs_to_save.items():
             if value is not None: f.attrs[key] = value
        logger.debug("Global attributes saved.")

        # --- VVVVV SAVE GROUND TRUTH BASEBAND VVVVV ---
        logger.info(f"Saving ground truth baseband signal ({len(baseband_signal_complex_ground_truth)} samples)...")
        f.create_dataset('ground_truth_baseband',
                         data=baseband_signal_complex_ground_truth,
                         compression='gzip')
        # Also store its sample rate as an attribute of the dataset for clarity
        f['ground_truth_baseband'].attrs['sample_rate_hz'] = fs_ground_truth
        logger.debug("Ground truth baseband saved.")
        # --- ^^^^^ SAVE GROUND TRUTH BASEBAND ^^^^^ ---

        # Store each chunk and its metadata
        logger.info(f"Saving {len(captured_chunks_data)} captured chunks...")
        for i_chunk, chunk_data in enumerate(captured_chunks_data):
            group_name = f'chunk_{i_chunk:03d}'
            # logger.debug(f"Saving group: {group_name}") # Can make log verbose
            group = f.create_group(group_name)
            group.create_dataset('iq_data', data=chunk_data, compression='gzip')
            if i_chunk < len(chunk_metadata):
                 # logger.debug(f"  Saving metadata for chunk {i_chunk}")
                 for key, value in chunk_metadata[i_chunk].items():
                     if value is not None: group.attrs[key] = value
            else:
                 logger.warning(f"Metadata missing for chunk index {i_chunk} during save.")
        logger.info("Chunk data saved.")

    logger.info("Data saved successfully.")
except Exception as e:
    logger.critical(f"\nError saving HDF5 file: {e}", exc_info=True)
    sys.exit(1)


# --- 5. Visualization (Example: Spectrum) ---
plot_results = False # Disable plotting in generator by default
if plot_results and captured_chunks_data:
    logger.info("Plotting spectrum...")
    # ... (Plotting code remains the same) ...
    plt.show()

logger.info("\nScript finished.")