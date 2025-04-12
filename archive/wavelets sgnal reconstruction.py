import numpy as np
import matplotlib.pyplot as plt
import pywt

# Parameters
sample_rate = 10e6  # 10 MHz sampling rate (Nyquist for 5 MHz max frequency)
duration = 0.001    # 1 ms duration
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False) # Time vector
noise_level = 0.5   # Noise amplitude

# Frequencies for the three signals (simulating downconverted IF)
freq_telstra = 1e6    # Telstra: 1 MHz
freq_optus = 2e6      # Optus: 2 MHz
freq_vodafone = 3e6   # Vodafone: 3 MHz

# Generate individual clean signals
telstra_signal = np.sin(2 * np.pi * freq_telstra * t)
optus_signal = 0.8 * np.sin(2 * np.pi * freq_optus * t) # Slightly different amplitudes
vodafone_signal = 1.2 * np.sin(2 * np.pi * freq_vodafone * t)

# Combine signals and add Gaussian noise
combined_signal = telstra_signal + optus_signal + vodafone_signal
noise = noise_level * np.random.normal(0, 1, len(t))
noisy_signal = combined_signal + noise

# --- Wavelet Processing ---

# Compute Continuous Wavelet Transform (CWT)
wavelet = 'cmor1.5-1.0' # Complex Morlet wavelet (good for frequency localization)
# Choose scales covering the 1-3 MHz range well within the 5 MHz Nyquist limit
scales = np.geomspace(sample_rate / (5 * freq_vodafone), sample_rate / (0.5 * freq_telstra), num=100) # Log scale range around 1-3 MHz
# Alternatively, use linear scales: scales = np.arange(1, 128)

coeffs, freqs = pywt.cwt(noisy_signal, scales, wavelet, sampling_period=1/sample_rate)
# coeffs will be complex-valued

# Find scales closest to the target frequencies
target_freqs = [freq_telstra, freq_optus, freq_vodafone]
scale_indices = [np.argmin(np.abs(freqs - f)) for f in target_freqs]

print(f"Target Frequencies (MHz): {[f/1e6 for f in target_freqs]}")
print(f"Closest Frequencies Found (MHz): {freqs[scale_indices] / 1e6}")
print(f"Corresponding Scales Indices: {scale_indices}")

# Reconstruct each signal using coefficients at the selected scale
# This is an approximation: taking the CWT coefficients at a specific scale
# isolates the signal component around that frequency.
reconstructed_signals = []
for i, scale_idx in enumerate(scale_indices):
    # The CWT coefficients are complex. The real part approximates the signal.
    # We might need scaling/normalization depending on wavelet energy
    # For simple visualization, the real part is often sufficient.
    reconstructed_signal = coeffs[scale_idx, :].real
    reconstructed_signals.append(reconstructed_signal)

# --- Plotting ---
plt.figure(figsize=(15, 10))
num_samples_to_plot = 300 # Plot more samples to see oscillations

# 1. Original Noisy Signal
plt.subplot(4, 1, 1)
plt.plot(t[:num_samples_to_plot] * 1e6, noisy_signal[:num_samples_to_plot], label='Noisy Combined Signal (IF)')
plt.title('Noisy Combined Signal (Input)')
plt.xlabel('Time (µs)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# 2. Reconstructed Telstra Signal (1 MHz)
plt.subplot(4, 1, 2)
# Plot reconstructed vs original clean signal for comparison
plt.plot(t[:num_samples_to_plot] * 1e6, reconstructed_signals[0][:num_samples_to_plot], label=f'Reconstructed Telstra ({freq_telstra/1e6:.1f} MHz)')
plt.plot(t[:num_samples_to_plot] * 1e6, telstra_signal[:num_samples_to_plot], '--', label='Original Telstra Signal', alpha=0.7)
plt.title('Reconstructed Telstra Signal')
plt.xlabel('Time (µs)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# 3. Reconstructed Optus Signal (2 MHz)
plt.subplot(4, 1, 3)
plt.plot(t[:num_samples_to_plot] * 1e6, reconstructed_signals[1][:num_samples_to_plot], label=f'Reconstructed Optus ({freq_optus/1e6:.1f} MHz)')
plt.plot(t[:num_samples_to_plot] * 1e6, optus_signal[:num_samples_to_plot], '--', label='Original Optus Signal', alpha=0.7)
plt.title('Reconstructed Optus Signal')
plt.xlabel('Time (µs)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# 4. Reconstructed Vodafone Signal (3 MHz)
plt.subplot(4, 1, 4)
plt.plot(t[:num_samples_to_plot] * 1e6, reconstructed_signals[2][:num_samples_to_plot], label=f'Reconstructed Vodafone ({freq_vodafone/1e6:.1f} MHz)')
plt.plot(t[:num_samples_to_plot] * 1e6, vodafone_signal[:num_samples_to_plot], '--', label='Original Vodafone Signal', alpha=0.7)
plt.title('Reconstructed Vodafone Signal')
plt.xlabel('Time (µs)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()