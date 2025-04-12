import numpy as np
from scipy.signal import butter, lfilter
from scipy.io.wavfile import write
from scipy import signal
import pywt

# Simulate FM radio signal with frequency modulation
fs = 44100.0
t = np.arange(0, 1, 1/fs)
f_mod = 1000  # carrier frequency
x_carrier = np.sin(2 * np.pi * f_mod * t)

f_modulation = 10  # modulation frequency (smaller than carrier freq.)
x_modulation = np.sin(2 * np.pi * f_modulation * t)
y_fm = x_carrier + 0.5*x_modulation

# Add noise with smaller standard deviation
noise_std = 100
y_noisy = y_fm + noise_std*np.random.randn(len(t))

# Apply low-pass filter with adjusted cutoff frequency and order
cutoff_frequency = 1500.0
order = 5
normal_cutoff = cutoff_frequency / (0.5 * fs)
b, a = butter(order, normal_cutoff, btype='low')
y_filtered = lfilter(b, a, y_noisy)

# Apply wavelet transform with more suitable basis (e.g., 'db4')
coeffs = pywt.dwt(y_filtered, 'db4')

# Plot original and transformed signals
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, y_noisy)
plt.title('Original FM Radio Signal with Noise')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(np.abs(coeffs[0]), label='Approximation Coefficients')
plt.legend()
plt.xlabel('Time (samples)')

plt.tight_layout()
plt.show()

# Write the filtered signal to a WAV file
write('filtered_fm_signal.wav', fs, y_filtered)
