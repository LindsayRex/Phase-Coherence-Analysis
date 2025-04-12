import numpy as np
import matplotlib.pyplot as plt
import pywt

# Parameters
sample_rate = 10e6  # 10 MHz sampling rate (arbitrary for simulation)
duration = 0.001    # 1 ms duration
t = np.linspace(0, duration, int(sample_rate * duration))  # Time vector
noise_level = 0.5   # Noise amplitude

# Simulate mobile phone signals (frequencies in Hz)
freq_telstra = 900e6   # Telstra: 900 MHz (e.g., 4G Band 8)
freq_optus = 1800e6    # Optus: 1800 MHz (e.g., 4G Band 3)
freq_vodafone = 2100e6 # Vodafone: 2100 MHz (e.g., 4G Band 1)

# Generate clean signals
telstra_signal = np.sin(2 * np.pi * freq_telstra * t)
optus_signal = np.sin(2 * np.pi * freq_optus * t)
vodafone_signal = np.sin(2 * np.pi * freq_vodafone * t)

# Combine signals
combined_signal = telstra_signal + optus_signal + vodafone_signal

# Add Gaussian noise
noise = noise_level * np.random.normal(0, 1, len(t))
noisy_signal = combined_signal + noise

# Wavelet transform parameters
wavelet = 'morl'  # Morlet wavelet (good for frequency localization)
scales = np.arange(1, 128)  # Scales to scan frequencies

# Perform Continuous Wavelet Transform (CWT)
coeffs, freqs = pywt.cwt(noisy_signal, scales, wavelet, sampling_period=1/sample_rate)

# Denoising: Thresholding wavelet coefficients
threshold = 0.1 * np.max(np.abs(coeffs))  # Simple threshold (10% of max)
coeffs_denoised = coeffs.copy()
coeffs_denoised[np.abs(coeffs_denoised) < threshold] = 0  # Zero out small coefficients

# Inverse CWT to reconstruct denoised signal
# Note: PyWavelets doesn't have a built-in ICWT, so we'll approximate by averaging large coefficients
denoised_signal = np.sum(coeffs_denoised, axis=0) / len(scales)  # Rough reconstruction

# Plotting
plt.figure(figsize=(15, 10))

# 1. Original Noisy Signal
plt.subplot(3, 1, 1)
plt.plot(t[:200], noisy_signal[:200], label='Noisy Signal (Telstra + Optus + Vodafone)')
plt.title('Noisy Combined Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

# 2. Wavelet Scalogram (CWT Coefficients)
plt.subplot(3, 1, 2)
plt.pcolormesh(t[:200], freqs / 1e6, np.abs(coeffs[:, :200]), shading='auto')
plt.colorbar(label='Magnitude')
plt.title('Wavelet Scalogram (CWT)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (MHz)')
plt.ylim(0, 2500)  # Limit to relevant frequency range

# 3. Denoised Signal
plt.subplot(3, 1, 3)
plt.plot(t[:200], denoised_signal[:200], label='Denoised Signal', color='orange')
plt.plot(t[:200], combined_signal[:200], '--', label='Original Clean Signal', color='green')
plt.title('Denoised Signal vs Original')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()

# Frequency detection (basic peak finding in CWT)
mean_coeffs = np.mean(np.abs(coeffs), axis=1)
detected_freqs = freqs[np.where(mean_coeffs > 0.5 * np.max(mean_coeffs))[0]]
print("Detected frequencies (MHz):", detected_freqs / 1e6)