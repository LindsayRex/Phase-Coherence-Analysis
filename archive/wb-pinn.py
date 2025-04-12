import numpy as np
import matplotlib.pyplot as plt
import pywt

# Parameters
sample_rate = 10e6  # 10 MHz sampling rate (Nyquist for 5 MHz max frequency)
duration = 0.001    # 1 ms duration
t = np.linspace(0, duration, int(sample_rate * duration))  # Time vector
noise_level = 0.5   # Noise amplitude

# Frequencies for the three signals
freq_telstra = 1e6    # Telstra: 1 MHz
freq_optus = 2e6      # Optus: 2 MHz
freq_vodafone = 3e6   # Vodafone: 3 MHz

# Generate individual signals
telstra_signal = np.sin(2 * np.pi * freq_telstra * t)
optus_signal = np.sin(2 * np.pi * freq_optus * t)
vodafone_signal = np.sin(2 * np.pi * freq_vodafone * t)

# Combine signals and add noise
combined_signal = telstra_signal + optus_signal + vodafone_signal
noisy_signal = combined_signal + noise_level * np.random.normal(0, 1, len(t))

# Compute Continuous Wavelet Transform (CWT)
wavelet = 'morl'  # Morlet wavelet for good frequency localization
scales = np.arange(1, 65)  # Scales from 1 to 64
coeffs, freqs = pywt.cwt(noisy_signal, scales, wavelet, sampling_period=1/sample_rate)

# Find scales corresponding to 1 MHz, 2 MHz, 3 MHz
target_freqs = [freq_telstra, freq_optus, freq_vodafone]
scale_indices = [np.argmin(np.abs(freqs - f)) for f in target_freqs]

# Reconstruct each signal using coefficients at the selected scale
reconstructed_signals = []
for scale_idx in scale_indices:
    # Use only the coefficients at the target scale
    reconstructed_signal = coeffs[scale_idx, :]
    reconstructed_signals.append(reconstructed_signal)

# Plotting
plt.figure(figsize=(12, 9))

# Reconstructed Telstra Signal (1 MHz)
plt.subplot(3, 1, 1)
plt.plot(t[:200], reconstructed_signals[0][:200], label='Reconstructed Telstra (1 MHz)')
plt.title('Reconstructed Telstra Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

# Reconstructed Optus Signal (2 MHz)
plt.subplot(3, 1, 2)
plt.plot(t[:200], reconstructed_signals[1][:200], label='Reconstructed Optus (2 MHz)')
plt.title('Reconstructed Optus Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

# Reconstructed Vodafone Signal (3 MHz)
plt.subplot(3, 1, 3)
plt.plot(t[:200], reconstructed_signals[2][:200], label='Reconstructed Vodafone (3 MHz)')
plt.title('Reconstructed Vodafone Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()