# utils.py
"""Utility functions."""

import numpy as np
import logging # Import standard logging

# --- Get logger for this module ---
logger = logging.getLogger(__name__)
# --- End Get logger ---
# DO NOT call log_config.setup_logging() here

def compute_spectrum(signal, sample_rate):
    """
    Compute the power spectrum of a signal in dB. Uses logging.

    Args:
        signal (np.ndarray): Input signal (complex or real).
        sample_rate (float): Sampling frequency in Hz.

    Returns:
        tuple: (freqs, spec_db)
               - freqs (np.ndarray): Frequency array (shifted for centered spectrum).
               - spec_db (np.ndarray): Power spectrum in dB (shifted). Returns empty if input invalid.
    """
    logger.debug(f"Computing spectrum: signal length={len(signal)}, sample_rate={sample_rate}")

    # Input validation
    if signal is None:
        logger.warning("Input signal is None. Cannot compute spectrum.")
        return np.array([]), np.array([])
    try:
        # Attempt conversion early to catch non-array types
        signal = np.asarray(signal)
    except Exception as e:
        logger.error(f"Could not convert input signal to NumPy array: {e}")
        return np.array([]), np.array([])

    n = len(signal)
    if n < 2:
        logger.warning(f"Cannot compute spectrum: Signal length {n} < 2.")
        return np.array([]), np.array([])
    if sample_rate is None or not isinstance(sample_rate, (int, float, np.number)) or sample_rate <= 0:
        logger.warning(f"Cannot compute spectrum: Invalid sample_rate={sample_rate} (type: {type(sample_rate)}).")
        return np.array([]), np.array([])

    # Ensure complex type and handle non-finite values
    try:
        signal_complex = signal.astype(np.complex128)
        finite_mask = np.isfinite(signal_complex)
        if not np.all(finite_mask):
            num_non_finite = np.sum(~finite_mask)
            logger.warning(f"Input signal contains {num_non_finite} non-finite values. Replacing with zeros.")
            signal_complex = np.nan_to_num(signal_complex) # Replaces NaN with 0, Inf with large numbers
            # Optional: More robust handling like interpolation or zeroing might be needed
    except Exception as e:
        logger.error(f"Error converting signal to complex or handling NaNs: {e}")
        return np.array([]), np.array([])


    try:
        # Compute FFT
        logger.debug(f"Computing FFT with length {n}...")
        fft_result = np.fft.fft(signal_complex)

        # Compute Power Spectrum = |FFT|^2 / n
        spec = (np.abs(fft_result)**2) / n
        logger.debug(f"Calculated power spectrum (min={np.min(spec):.2e}, max={np.max(spec):.2e})")

        # Convert to dB, adding epsilon to avoid log(0)
        epsilon = 1e-20 # Small value to prevent log10(0)
        spec_db = 10 * np.log10(spec + epsilon)

        # Generate frequency axis
        freqs = np.fft.fftfreq(n, d=1/sample_rate)

        # Shift both frequency and spectrum for plotting centered at 0 Hz
        logger.debug("FFT shifting complete.")
        return np.fft.fftshift(freqs), np.fft.fftshift(spec_db)

    except Exception as spec_e:
        logger.error(f"Error during FFT or spectrum calculation: {spec_e}", exc_info=True)
        return np.array([]), np.array([])