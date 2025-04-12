# generate_sequential_chunks.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import h5py
from tqdm import tqdm
import sys
import os
import json
import time

# --- Configuration ---
CONFIG_CHOICE = 'perfect_clock' # Options: 'perfect_clock', 'random_phase_jump'
OUTPUT_SUFFIX = '_PERFECT_CLOCK' if CONFIG_CHOICE == 'perfect_clock' else '_RANDOM_PHASE'

# --- Simulation Parameters (Similar to your original script) ---
f_rf_center = 25e9
bw_total_signal = 400e6
modulation = 'qam16'
sdr_ibw = 56e6
adc_bits = 12
adc_vref = 1.0
num_chunks = 8
overlap_factor = 0.1
snr_db = 25 # SNR per chunk
duration_per_chunk = 100e-6
tuning_delay = 5e-6

# --- Parameters specifically for the synthetic generator ---
# These might map to the JSON config of the original generator
# We'll use fs_sdr derived below as the sample rate for generation
symbol_rate_factor = 0.5 # Example: Symbol rate = 0.5 * sdr_ibw for QAM16
tx_filter_type = 'rrc' # Root-raised cosine for QAM
tx_filter_beta = 0.3   # Excess bandwidth for RRC
tx_filter_symbols = 12 # Filter delay in symbols

# --- Derived Parameters ---
chunk_spacing = sdr_ibw * (1.0 - overlap_factor)
actual_num_chunks_needed = int(np.ceil((bw_total_signal - sdr_ibw) / chunk_spacing)) + 1
if actual_num_chunks_needed > num_chunks:
    print(f"Warning: {num_chunks} chunks might not cover full {bw_total_signal/1e6} MHz. Need {actual_num_chunks_needed}. Using {num_chunks}.")

total_covered_bw = chunk_spacing * (num_chunks - 1) + sdr_ibw
start_rf_chunk_center = f_rf_center - (chunk_spacing * (num_chunks - 1)) / 2
rf_chunk_centers = np.linspace(start_rf_chunk_center,
                               start_rf_chunk_center + chunk_spacing * (num_chunks - 1),
                               num_chunks)

# Determine SDR sample rate (maybe fixed by generator's underlying process)
# Let's assume the generator creates data at a rate related to IBW, e.g. 2x Nyquist
fs_sdr = sdr_ibw * 2.0 # Sample rate matching generator's likely output
num_samples_per_chunk = int(round(duration_per_chunk * fs_sdr))
print(f"Target samples per chunk: {num_samples_per_chunk} at {fs_sdr/1e6:.2f} MHz")

# Phase noise model parameter (used if random jumps are simulated)
phase_noise_std_dev_rad = np.deg2rad(1.0) if CONFIG_CHOICE == 'random_phase_jump' else 0.0


print(f"\n--- Parameters Summary ---")
print(f"Simulation Type: {CONFIG_CHOICE}")
print(f"Total Signal BW: {bw_total_signal / 1e6:.1f} MHz @ RF Center {f_rf_center / 1e9:.2f} GHz")
print(f"SDR IBW: {sdr_ibw / 1e6:.1f} MHz")
print(f"SDR Sample Rate (Generator): {fs_sdr / 1e6:.1f} MHz")
print(f"Number of Chunks: {num_chunks}")
print(f"Duration per Chunk: {duration_per_chunk * 1e6:.1f} µs")
print(f"Samples per Chunk: {num_samples_per_chunk}")
print(f"Tuning Delay: {tuning_delay * 1e6:.1f} µs")
print(f"Simulated Inter-Chunk Phase Noise Std Dev: {np.rad2deg(phase_noise_std_dev_rad):.1f} deg")
print(f"SNR per Chunk: {snr_db} dB")
print(f"ADC Bits: {adc_bits}")
print(f"RF Chunk Centers (GHz): {[f / 1e9:.4f for f in rf_chunk_centers]}")
print(f"--------------------------\n")


# --- Placeholder for the Core Generation Logic ---
# You need to adapt this function based on the actual code in
# https://github.com/TorchEnsemble/synthetic-rf-dataset-generator/blob/main/generator.py
# and potentially its utils. It should use liquid-dsp wrappers.

def generate_rf_chunk(n_samps, modulation_type, symbol_rate,
                      tx_filter_params, snr, freq_offset, phase_offset):
    """
    Placeholder function to generate one chunk of RF data using synthetic generator logic.

    Args:
        n_samps (int): Number of samples to generate.
        modulation_type (str): e.g., 'qam16'.
        symbol_rate (float): Symbol rate in samples/symbol (relative to fs_sdr).
        tx_filter_params (dict): Dictionary with 'type', 'beta', 'delay'.
        snr (float): Signal-to-Noise Ratio in dB.
        freq_offset (float): Normalized frequency offset (0 to 0.5).
        phase_offset (float): Phase offset in radians.

    Returns:
        np.ndarray: Complex IQ data for the chunk.
    """
    print(f"  Generating chunk: {n_samps} samples, Mod: {modulation_type}, SR: {symbol_rate:.2f} Sps, "
          f"SNR: {snr:.1f} dB, FO: {freq_offset:.4f}, PO: {np.rad2deg(phase_offset):.2f} deg")

    # --- THIS IS WHERE YOU INTEGRATE THE CORE LOGIC FROM THE GENERATOR REPO ---
    # Example using dummy data - REPLACE THIS
    if modulation_type == 'qam16':
        # Simplified generation - does not use liquid-dsp or proper filtering
        num_symbols = int(np.ceil(n_samps / symbol_rate))
        symbols = (np.random.choice([-3,-1,1,3], size=num_symbols) +
                   1j*np.random.choice([-3,-1,1,3], size=num_symbols)) / np.sqrt(10)
        # Basic pulse shaping (repeat symbols) - REPLACE with proper filtering
        iq_data = np.repeat(symbols, int(round(symbol_rate)))
        # Trim/pad
        if len(iq_data) > n_samps: iq_data = iq_data[:n_samps]
        elif len(iq_data) < n_samps: iq_data = np.pad(iq_data, (0, n_samps - len(iq_data)))

    elif modulation_type == 'noise':
        iq_data = np.random.randn(n_samps) + 1j*np.random.randn(n_samps)
        # Apply BW limit if needed (e.g. filter) - REPLACE with proper filtering
    else:
        iq_data = np.zeros(n_samps, dtype=np.complex128)

    # Normalize signal power before adding noise (generator might do this)
    sig_power = np.mean(np.abs(iq_data)**2)
    if sig_power > 1e-15:
        iq_data = iq_data / np.sqrt(sig_power) # Normalize to approx 0dB power

    # Add AWGN
    signal_power = 1.0 # Since we normalized
    snr_linear = 10**(snr / 10.0)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power / 2.0)
    noise = noise_std * (np.random.randn(n_samps) + 1j*np.random.randn(n_samps))
    iq_noisy = iq_data + noise

    # Apply offsets (Generator likely applies these internally)
    time_vec = np.arange(n_samps) / fs_sdr # Use fs_sdr here
    iq_offset = iq_noisy * np.exp(1j * (2 * np.pi * freq_offset * fs_sdr * time_vec + phase_offset))

    print(f"  Dummy generation complete. RMS={np.sqrt(np.mean(np.abs(iq_offset)**2)):.3e}")
    # Replace with actual call to the generator's functions using liquid-dsp
    # e.g., import generate from generator; iq_offset = generate(...)
    # You'll need to pass parameters appropriately.
    # --- END OF PLACEHOLDER ---

    return iq_offset


# --- Simulate Sequential Chunk Capture ---
print(f"\nSimulating capture of {num_chunks} chunks using synthetic generator logic...")
captured_chunks_data = []
chunk_metadata = []
cumulative_phase_offset_sim = 0.0 # Track the simulated offset

# Calculate symbol rate for the generator based on fs_sdr
sdr_symbol_rate_sps = symbol_rate_factor * sdr_ibw # Symbols per second
sdr_symbol_rate_samples = fs_sdr / sdr_symbol_rate_sps # Samples per symbol

tx_filter_config = {
    "type": tx_filter_type,
    "beta": tx_filter_beta,
    "delay": tx_filter_symbols
}

for i in tqdm(range(num_chunks)):
    current_rf_center = rf_chunk_centers[i]
    # The generator likely works at baseband, centered at 0 Hz.
    # Frequency/Phase offsets simulate imperfections relative to this center.

    # --- Calculate Phase Offset for this Chunk ---
    if i == 0:
        current_phase_offset = 0.0 # First chunk is reference
    else:
        if CONFIG_CHOICE == 'random_phase_jump':
            # Add random jump based on std dev
            phase_jump = np.random.randn() * phase_noise_std_dev_rad
            cumulative_phase_offset_sim += phase_jump
            current_phase_offset = cumulative_phase_offset_sim
            print(f"  Chunk {i}: Applying random phase jump = {np.rad2deg(phase_jump):.4f} deg")
        else: # perfect_clock
            # Assume zero relative phase jump between chunks
            current_phase_offset = 0.0
            # If you had small known metadata offsets, apply them here instead
            # e.g., current_phase_offset = known_metadata_offsets[i]

    # --- Generate Chunk using Placeholder/Adapted Function ---
    # Frequency offset is typically small, simulate zero for now
    freq_offset_normalized = 0.0

    chunk_iq_data = generate_rf_chunk(
        n_samps=num_samples_per_chunk,
        modulation_type=modulation,
        symbol_rate=sdr_symbol_rate_samples, # Pass samples/symbol
        tx_filter_params=tx_filter_config,
        snr=snr_db,
        freq_offset=freq_offset_normalized, # Normalized freq offset
        phase_offset=current_phase_offset   # Apply calculated phase offset
    )

    # --- Simulate ADC Quantization ---
    max_abs = np.max(np.abs(chunk_iq_data)) if len(chunk_iq_data) > 0 else 1.0
    if max_abs < 1e-15: max_abs = 1.0
    scale_adc = (adc_vref / 2.0) / (max_abs * 1.1) # Dynamic range scaling
    quant_step = adc_vref / (2**adc_bits)
    quantized_real = np.round((chunk_iq_data.real * scale_adc) / quant_step) * quant_step
    quantized_imag = np.round((chunk_iq_data.imag * scale_adc) / quant_step) * quant_step
    quantized_real = np.clip(quantized_real, -adc_vref/2 + quant_step/2, adc_vref/2 - quant_step/2)
    quantized_imag = np.clip(quantized_imag, -adc_vref/2 + quant_step/2, adc_vref/2 - quant_step/2)
    final_chunk_data = (quantized_real + 1j*quantized_imag) / scale_adc # Rescale back

    # --- Store Chunk Data and Metadata ---
    captured_chunks_data.append(final_chunk_data)
    meta = {
        'chunk_index': i,
        'rf_center_freq_hz': current_rf_center, # Still relevant conceptually
        'sdr_sample_rate_hz': fs_sdr,
        'num_samples': len(final_chunk_data),
        'intended_duration_s': duration_per_chunk,
        'start_time_s': i * (duration_per_chunk + tuning_delay), # Approximate start time
        'applied_phase_offset_rad': current_phase_offset, # Store the offset used
        'snr_db': snr_db,
        'modulation': modulation,
        'tx_filter': json.dumps(tx_filter_config) # Store filter params
    }
    chunk_metadata.append(meta)

print("\nChunk capture simulation complete.")

# --- 4. Data Storage (Using HDF5) ---
script_dir = os.path.dirname(__file__) if "__file__" in locals() else "."
output_dir = os.path.abspath(os.path.join(script_dir, '..', 'Simulated_data'))
os.makedirs(output_dir, exist_ok=True)
output_filename_base = f"synthetic_gen_{f_rf_center / 1e9:.0f}GHz_{bw_total_signal / 1e6:.0f}MHzBW_{modulation}_sdr{sdr_ibw / 1e6:.0f}MHz"
output_filename = os.path.join(output_dir, f"{output_filename_base}{OUTPUT_SUFFIX}.h5")

print(f"Saving chunk data and metadata to: {output_filename}")
try:
    with h5py.File(output_filename, 'w') as f:
        # Store global attributes
        f.attrs['rf_center_freq_hz'] = f_rf_center
        f.attrs['total_signal_bandwidth_hz'] = bw_total_signal
        f.attrs['modulation'] = modulation
        f.attrs['sdr_ibw_hz'] = sdr_ibw
        f.attrs['sdr_sample_rate_hz'] = fs_sdr # Rate used by generator
        f.attrs['num_chunks'] = num_chunks
        f.attrs['intended_num_chunks'] = num_chunks
        f.attrs['overlap_factor'] = overlap_factor
        f.attrs['snr_db_per_chunk'] = snr_db
        f.attrs['adc_bits'] = adc_bits
        f.attrs['tuning_delay_s'] = tuning_delay
        f.attrs['phase_offset_std_dev_rad_simulated'] = phase_noise_std_dev_rad
        f.attrs['simulation_type'] = CONFIG_CHOICE
        # Store ground truth rate info if needed for reconstruction later
        # f.attrs['ground_truth_sample_rate_hz'] = fs_ground_truth_used_in_recon

        f.attrs['actual_num_chunks_saved'] = len(captured_chunks_data)
        f.attrs['tx_filter_params'] = json.dumps(tx_filter_config)

        for i_chunk, chunk_data in enumerate(captured_chunks_data):
            group_name = f'chunk_{i_chunk:03d}'
            group = f.create_group(group_name)
            group.create_dataset('iq_data', data=chunk_data, compression='gzip')
            if i_chunk < len(chunk_metadata):
                 for key, value in chunk_metadata[i_chunk].items():
                     group.attrs[key] = value

    print("Data saved successfully.")
except Exception as e:
    print(f"\nError saving HDF5 file: {e}")
    sys.exit(1)


# --- 5. Visualization (Spectrum Plot) ---
plot_results = True
if plot_results and captured_chunks_data:
    print("Plotting spectrum...")
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(14, 8))
    max_overall_magnitude = 0.0
    num_plotted = 0

    for i_chunk, chunk_data in enumerate(captured_chunks_data):
        if chunk_data is not None and len(chunk_data) >= 2 and np.all(np.isfinite(chunk_data)):
             mag = np.max(np.abs(chunk_data))
             if mag > max_overall_magnitude: max_overall_magnitude = mag
             num_plotted += 1
        else: print(f"Warning: Skipping spectrum plot for chunk {i_chunk} due to invalid data.")

    if num_plotted == 0: print("No valid chunks to plot spectrum.")
    if max_overall_magnitude < 1e-15: max_overall_magnitude = 1.0

    if num_plotted > 0:
         for i_chunk, chunk_data in enumerate(captured_chunks_data):
             if chunk_data is None or len(chunk_data) < 2 or not np.all(np.isfinite(chunk_data)): continue
             if i_chunk < len(chunk_metadata): meta = chunk_metadata[i_chunk]
             else: continue

             n_fft = len(chunk_data)
             # Frequency axis relative to the chunk's RF center
             freqs_baseband = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1 / meta['sdr_sample_rate_hz']))
             rf_freqs = freqs_baseband + meta['rf_center_freq_hz']
             spectrum = np.fft.fftshift(np.fft.fft(chunk_data))
             # Normalize spectrum magnitude by N*max_overall_magnitude
             spectrum_db = 20 * np.log10(np.abs(spectrum) / (n_fft * max_overall_magnitude) + 1e-12)

             plt.plot(rf_freqs / 1e9, spectrum_db, label=f'Chunk {i_chunk} (RF: {meta["rf_center_freq_hz"] / 1e9:.3f} GHz)', alpha=0.8)

         plt.title(f'Spectra of Captured Chunks ({CONFIG_CHOICE}, Shifted to RF)\nMod: {modulation}, SNR: {snr_db} dB, SDR Rate: {fs_sdr/1e6:.1f} MHz')
         plt.xlabel('Frequency (GHz)')
         plt.ylabel('Magnitude (dB rel. Peak Time Signal)')
         plt.ylim(bottom=-140)
         plt.grid(True, which='both', linestyle='--')
         if len(captured_chunks_data) > 8: plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize='small'); plt.tight_layout(rect=[0, 0, 0.85, 1])
         else: plt.legend(fontsize='small'); plt.tight_layout()
         plt.show()

print("\nScript finished.")