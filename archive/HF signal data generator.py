import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from tqdm import tqdm # Optional: for progress bar on large simulations

# --- Simulation Parameters ---
# RF Stage
f_rf = 25e9  # Example RF Carrier Frequency (25 GHz) - SET YOUR TARGET
bw_signal = 10e6 # Signal Bandwidth (e.g., 10 MHz for BPSK)
modulation = 'bpsk' # 'bpsk' or 'cw' (Continuous Wave)

# Downconversion Stage
f_lo = 24.9e9 # Local Oscillator Frequency (e.g., 24.9 GHz)
f_if = f_rf - f_lo # Resulting Intermediate Frequency (100 MHz in this example)
if f_if <= 0:
    raise ValueError(f"LO frequency ({f_lo/1e9} GHz) must be less than RF frequency ({f_rf/1e9} GHz)")

# IF / ADC Stage
# Need to sample based on IF frequency AND signal bandwidth
# Nyquist for IF signal = 2 * (f_if + bw_signal / 2)
fs_if_nyquist = 2 * (f_if + bw_signal / 2)
oversample_factor = 2.5 # Factor > 1 for easier filtering, processing. E.g., 2.5
fs_if = fs_if_nyquist * oversample_factor # IF sampling rate (e.g., 2.5 * (100MHz + 5MHz) = 262.5 MHz)

adc_bits = 12 # Number of ADC quantization bits
adc_vref = 1.0 # Voltage range of ADC (+/- Vref/2 assumed, so full range is Vref)

# Channel Stage
snr_db = 15 # Signal-to-Noise Ratio (in dB)

# Simulation Time / Signal Definition
symbol_rate = bw_signal # For simple BPSK, symbol rate approx equals bandwidth
duration = 1000 / symbol_rate # Simulate 1000 symbols duration
num_samples_if = int(duration * fs_if)
t_if = np.linspace(0, duration, num_samples_if, endpoint=False) # Time vector at IF rate

print(f"--- Parameters ---")
print(f"RF Frequency: {f_rf/1e9:.2f} GHz")
print(f"Signal Bandwidth: {bw_signal/1e6:.2f} MHz")
print(f"LO Frequency: {f_lo/1e9:.2f} GHz")
print(f"IF Frequency: {f_if/1e6:.2f} MHz")
print(f"IF Nyquist Rate: {fs_if_nyquist/1e6:.2f} MHz")
print(f"IF Actual Sample Rate: {fs_if/1e6:.2f} MHz (Oversample: {oversample_factor}x)")
print(f"Symbol Rate: {symbol_rate/1e6:.2f} Msps")
print(f"Simulation Duration: {duration*1e6:.2f} µs")
print(f"Number of IF Samples: {num_samples_if}")
print(f"SNR: {snr_db} dB")
print(f"ADC Bits: {adc_bits}")
print(f"------------------\n")

# --- 1. Generate Baseband Signal ---
if modulation.lower() == 'bpsk':
    num_symbols = int(duration * symbol_rate)
    # Generate random symbols (+1, -1)
    symbols = np.random.choice([-1, 1], size=num_symbols)
    # Create symbol stream at IF sample rate (simple rectangular pulses)
    samples_per_symbol = int(fs_if / symbol_rate)
    baseband_symbols = np.repeat(symbols, samples_per_symbol)
    # Ensure length matches num_samples_if (trim or pad if needed)
    baseband_symbols = baseband_symbols[:num_samples_if]
    # Normalize amplitude (optional, helps with power calcs later)
    baseband_signal = baseband_symbols / np.sqrt(np.mean(baseband_symbols**2))
    # BPSK phase is 0 or pi, so signal is real here. For QPSK it would be complex.
    baseband_signal_complex = baseband_signal # Keep as complex type for consistency
    print(f"Generated {num_symbols} BPSK symbols.")

elif modulation.lower() == 'cw':
    baseband_signal_complex = np.ones(num_samples_if, dtype=complex) # Constant amplitude/phase
    print("Generated CW signal.")
else:
    raise ValueError(f"Unsupported modulation: {modulation}")

# Optional: Add pulse shaping (e.g., Root Raised Cosine) here for better BW control
# ... (implementation requires RRC filter design)

# --- 2. Simulate Upconversion to RF (Mathematically) ---
# We don't explicitly create the massive RF time vector.
# We calculate the phase corresponding to RF at the IF sample times.
# phase_rf = 2 * np.pi * f_rf * t_if
# signal_rf_complex = baseband_signal_complex * np.exp(1j * phase_rf)
# This step is conceptual - we directly apply downconversion next.

# --- 3. Simulate Downconversion (Mixing + Filtering) ---
print("Simulating downconversion...")
# Mix with LO: multiply by complex conjugate of LO signal
# sig_mixed = signal_rf_complex * np.exp(-1j * 2 * np.pi * f_lo * t_if)
# Combining upconversion and downconversion:
# sig_mixed = (baseband * exp(j*2*pi*f_rf*t)) * exp(-j*2*pi*f_lo*t)
# sig_mixed = baseband * exp(j*2*pi*(f_rf - f_lo)*t) = baseband * exp(j*2*pi*f_if*t)
# This shows mixing RF down results in baseband modulated onto the IF carrier
if_signal_carrier = baseband_signal_complex * np.exp(1j * 2 * np.pi * f_if * t_if)

# Design Low-Pass Filter (Butterworth example) to isolate IF
# Cutoff slightly above the signal bandwidth around the IF
lpf_cutoff = bw_signal * 0.6 # Cutoff freq passband edge (e.g., 60% of BW)
# Ensure cutoff is reasonable compared to fs_if
if lpf_cutoff >= fs_if / 2:
     lpf_cutoff = fs_if / 2.1 # Avoid numerical instability near Nyquist
     print(f"Warning: LPF cutoff adjusted to {lpf_cutoff/1e6:.2f} MHz")

filter_order = 5 # Filter order
b, a = sig.butter(filter_order, lpf_cutoff, btype='low', analog=False, fs=fs_if)

# Apply filter
if_signal_filtered = sig.lfilter(b, a, if_signal_carrier)
# Use filtfilt for zero phase distortion (better for simulation, non-causal)
# if_signal_filtered = sig.filtfilt(b, a, if_signal_carrier)

# --- 4. Simulate Channel Effects (AWGN) ---
print("Adding AWGN...")
# Calculate signal power AFTER filtering
signal_power = np.mean(np.abs(if_signal_filtered)**2)
# Calculate noise power based on SNR
snr_linear = 10**(snr_db / 10.0)
noise_power = signal_power / snr_linear
# Calculate noise standard deviation (for complex noise)
noise_std_dev = np.sqrt(noise_power / 2.0)

# Generate complex Gaussian noise
noise_real = noise_std_dev * np.random.randn(num_samples_if)
noise_imag = noise_std_dev * np.random.randn(num_samples_if)
complex_noise = noise_real + 1j * noise_imag

# Add noise to signal
if_signal_noisy = if_signal_filtered + complex_noise

# --- 5. Simulate ADC Effects (Quantization) ---
print("Simulating ADC quantization...")
# Determine signal range for scaling (use noisy signal range)
# Consider real and imaginary parts separately
max_abs_val = np.max([np.max(np.abs(if_signal_noisy.real)), np.max(np.abs(if_signal_noisy.imag))])
# Scale factor to fit into ADC range (+/- adc_vref/2)
# Add a small margin (e.g., 10%) to prevent clipping often
scale_factor = (adc_vref / 2.0) / (max_abs_val * 1.1)

signal_scaled_real = if_signal_noisy.real * scale_factor
signal_scaled_imag = if_signal_noisy.imag * scale_factor

# Quantize
num_levels = 2**adc_bits
quant_step = adc_vref / num_levels

# Quantize real and imaginary parts
quantized_real = np.round(signal_scaled_real / quant_step) * quant_step
quantized_imag = np.round(signal_scaled_imag / quant_step) * quant_step

# Clip any values that might have exceeded the range slightly due to rounding/scaling
quantized_real = np.clip(quantized_real, -adc_vref/2, adc_vref/2)
quantized_imag = np.clip(quantized_imag, -adc_vref/2, adc_vref/2)

# Combine back into complex signal (still scaled)
if_quantized_scaled = quantized_real + 1j * quantized_imag

# Optional: Rescale back to original signal level (may not be necessary for PINN input)
if_quantized = if_quantized_scaled / scale_factor

# --- 6. Data Output & Visualization ---
final_output_signal = if_quantized # This is the data your PINN would receive

print(f"\nSimulation Complete. Final signal shape: {final_output_signal.shape}")

# --- Plotting (Optional - plot selected stages) ---
plot_stages = True
if plot_stages:
    print("Plotting results...")
    plt.style.use('seaborn-v0_8-darkgrid') # Nicer plots
    num_plot_samples = min(500, num_samples_if) # Plot fewer samples for clarity

    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Plot Baseband (if BPSK)
    if modulation.lower() == 'bpsk':
         axs[0].plot(t_if[:num_plot_samples]*1e6, baseband_signal_complex[:num_plot_samples].real, label='Baseband Symbols (Real)')
         axs[0].set_title('Baseband Signal (Real Part)')
         axs[0].set_ylabel('Amplitude')
         axs[0].legend()
         axs[0].grid(True)
    else:
         axs[0].plot(t_if[:num_plot_samples]*1e6, baseband_signal_complex[:num_plot_samples].real, label='Baseband (Real)')
         axs[0].set_title('Baseband Signal (CW)')
         axs[0].set_ylabel('Amplitude')
         axs[0].legend()
         axs[0].grid(True)


    # Plot Filtered IF Signal
    axs[1].plot(t_if[:num_plot_samples]*1e6, if_signal_filtered[:num_plot_samples].real, label='IF Filtered (Real)')
    axs[1].set_title(f'Filtered IF Signal ({f_if/1e6:.1f} MHz)')
    axs[1].set_ylabel('Amplitude')
    axs[1].legend()
    axs[1].grid(True)

    # Plot Noisy IF Signal
    axs[2].plot(t_if[:num_plot_samples]*1e6, if_signal_noisy[:num_plot_samples].real, label='IF Noisy (Real)', alpha=0.8)
    axs[2].set_title(f'Noisy IF Signal (SNR ~{snr_db} dB)')
    axs[2].set_ylabel('Amplitude')
    axs[2].legend()
    axs[2].grid(True)

    # Plot Quantized vs Noisy IF Signal (Zoomed)
    axs[3].plot(t_if[:num_plot_samples]*1e6, if_signal_noisy[:num_plot_samples].real, label='Noisy (Real)', alpha=0.6)
    axs[3].plot(t_if[:num_plot_samples]*1e6, if_quantized[:num_plot_samples].real, label=f'Quantized ({adc_bits}-bit) (Real)', linestyle='--')
    axs[3].set_title('Quantized vs. Noisy Signal (Real Part)')
    axs[3].set_xlabel('Time (µs)')
    axs[3].set_ylabel('Amplitude')
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()

    # Plot Spectrum
    plt.figure(figsize=(10, 6))
    f_axis = np.fft.fftshift(np.fft.fftfreq(num_samples_if, d=1/fs_if))
    # Spectrum of Filtered IF
    spectrum_filt = np.fft.fftshift(np.fft.fft(if_signal_filtered))
    plt.plot(f_axis / 1e6, 20 * np.log10(np.abs(spectrum_filt) / np.max(np.abs(spectrum_filt)) + 1e-12), label='Filtered IF Spectrum', alpha=0.7)
     # Spectrum of Noisy IF
    spectrum_noisy = np.fft.fftshift(np.fft.fft(if_signal_noisy))
    plt.plot(f_axis / 1e6, 20 * np.log10(np.abs(spectrum_noisy) / np.max(np.abs(spectrum_noisy)) + 1e-12), label='Noisy IF Spectrum', alpha=0.7)
    # Spectrum of Quantized IF
    spectrum_quant = np.fft.fftshift(np.fft.fft(if_quantized))
    plt.plot(f_axis / 1e6, 20 * np.log10(np.abs(spectrum_quant) / np.max(np.abs(spectrum_quant)) + 1e-12), label='Quantized IF Spectrum', linestyle=':')

    plt.title('Frequency Spectrum at IF Stage')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Magnitude (dB rel. Max)')
    plt.ylim(-80, 5) # Adjust Y limits if needed
    plt.legend()
    plt.grid(True)
    plt.show()

# --- 7. Data Storage Discussion ---

"""
Should I store the data? Is a DB necessary or is it just wave files?

Data Storage Options:

1.  Wave Files (.wav):
    *   Pros: Standard audio format.
    *   Cons: Primarily for audio (real-valued, typically 1-2 channels). NOT suitable for complex IQ data. Storing real/imag as separate channels is non-standard and loses the complex relationship for processing tools. AVOID for IQ data.

2.  Simple Binary Files (.dat, .iq, .cfile):
    *   Pros: Compact, efficient, widely used in SDR (GNU Radio File Sink/Source). Directly maps to memory layout (e.g., `complex64`). Fast read/write for sequential access.
    *   Cons: No built-in metadata. You MUST store sample rate, center frequency, data type (e.g., `complex64`, `int16`) separately (e.g., in filename or a separate .hdr file).
    *   How: `final_output_signal.astype(np.complex64).tofile('signal_25GHz_10MHzBW_100MHzIF_15dB_12bit.iq')`

3.  NumPy Arrays (.npy, .npz):
    *   Pros: Very easy in Python (`np.save`, `np.load`). Stores array structure and data type. `.npz` can store multiple arrays and potentially small metadata dicts in one compressed file.
    *   Cons: Python-specific (though readable by other languages with libraries). Might be less efficient for extremely large files compared to pure binary streaming.
    *   How: `np.save('signal_run_01.npy', final_output_signal)` or `np.savez('signal_run_01.npz', iq_data=final_output_signal, sample_rate=fs_if, center_freq=f_if)`

4.  HDF5 (.h5, .hdf5):
    *   Pros: Hierarchical Data Format. Excellent for large/complex datasets. Stores arrays, metadata, attributes in a structured way. Supports chunking and compression. Cross-platform. Good for organizing many simulation runs with parameters.
    *   Cons: Requires an extra library (`h5py`). Slightly more complex setup than `.npy`.
    *   How: (Requires `pip install h5py`)
        ```python
        # import h5py
        # with h5py.File('simulation_data.h5', 'a') as f: # 'a' appends or creates
        #     group = f.create_group('run_01_25GHz_BPSK')
        #     group.create_dataset('iq_data', data=final_output_signal, compression='gzip')
        #     group.attrs['sample_rate_hz'] = fs_if
        #     group.attrs['if_center_freq_hz'] = f_if
        #     group.attrs['rf_center_freq_hz'] = f_rf
        #     group.attrs['modulation'] = modulation
        #     group.attrs['snr_db'] = snr_db
        #     group.attrs['adc_bits'] = adc_bits
        ```

5.  Database (SQL, NoSQL):
    *   Pros: Powerful querying, indexing, data management features.
    *   Cons: Significant overhead for storing raw time-series IQ data. Read/write performance for bulk samples is much worse than file-based methods. Primarily useful for storing METADATA about simulations (parameters, results) or maybe highly processed FEATURES, not the raw IQ stream itself. OVERKILL for this purpose.

**Recommendation:**

*   **For individual simulation runs:** Use **`.npy`** (easiest in Python) or **simple binary (`.iq`, `.dat`)** if you need compatibility with tools like GNU Radio. Remember to save metadata separately for binary files.
*   **For managing many simulation runs with parameters:** Use **HDF5 (`.h5`)**. It's the most robust and scalable solution for organizing scientific data.
*   **Avoid `.wav`** for IQ data.
*   Use a **database only for metadata** or extracted results, not the bulk IQ samples.

"""

# Example saving using NumPy .npz
# Construct metadata dictionary
metadata = {
    'rf_center_freq_hz': f_rf,
    'lo_freq_hz': f_lo,
    'if_center_freq_hz': f_if,
    'signal_bandwidth_hz': bw_signal,
    'if_sample_rate_hz': fs_if,
    'modulation': modulation,
    'snr_db': snr_db,
    'adc_bits': adc_bits,
    'duration_s': duration,
    'num_samples': num_samples_if,
    'data_type': str(final_output_signal.dtype)
}
# filename = f"simulated_iq_{f_rf/1e9:.0f}GHz_{modulation}_if{f_if/1e6:.0f}MHz_snr{snr_db}dB.npz"
# print(f"\nSaving data and metadata to: {filename}")
# np.savez_compressed(filename, iq_data=final_output_signal, metadata=metadata)
# print("Data saved.")

# To load later:
# data = np.load(filename, allow_pickle=True) # allow_pickle for metadata dict
# loaded_iq = data['iq_data']
# loaded_metadata = data['metadata'].item() # .item() to get dict back
# print("\nLoaded Metadata Example:")
# print(loaded_metadata)