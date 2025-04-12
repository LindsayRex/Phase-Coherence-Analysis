# utils.py
"""Utility functions."""

import numpy as np
import logging
from . import log_config

# Setup logging for this module
log_config.setup_logging(level=logging.DEBUG, log_dir="run_logs")
logger = logging.getLogger(__name__)

def compute_spectrum(signal, sample_rate):
    """
    Compute the power spectrum of a signal in dB.

    Args:
        signal (np.ndarray): Input signal (complex or real).
        sample_rate (float): Sampling frequency in Hz.

    Returns:
        tuple: (freqs, spec_db)
               - freqs (np.ndarray): Frequency array (shifted for centered spectrum).
               - spec_db (np.ndarray): Power spectrum in dB (shifted). Returns empty if input invalid.
    """
    signal = np.asarray(signal)
    n = len(signal)
    if n < 2 or sample_rate is None or sample_rate <= 0:
        logger.warning(f"Cannot compute spectrum with n={n}, sample_rate={sample_rate}")
        return np.array([]), np.array([])

    signal_complex = signal.astype(np.complex128)
    if not np.all(np.isfinite(signal_complex)):
        logger.warning("Non-finite values in signal for spectrum computation. Replacing with zeros.")
        signal_complex = np.nan_to_num(signal_complex)
    try:
        fft_result = np.fft.fft(signal_complex)
        # Power Spectrum = |FFT|^2 / n
        spec = (np.abs(fft_result)**2) / n
        # Add epsilon slightly larger than machine epsilon for float64
        spec_db = 10 * np.log10(spec + 1e-20)
        freqs = np.fft.fftfreq(n, d=1/sample_rate)
        return np.fft.fftshift(freqs), np.fft.fftshift(spec_db)
    except Exception as spec_e:
        logger.error(f"Error computing spectrum: {spec_e}")
        return np.array([]), np.array([])