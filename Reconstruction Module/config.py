# config.py
"""Configuration parameters for the signal stitching pipeline."""
import os

# --- Input File ---
# Make sure this points to the HDF5 file generated WITH the pilot tone
# AND containing the ground_truth_baseband dataset.
INPUT_FILENAME = "simulated_chunks_25GHz_400MHzBW_qam16_sdr56MHz.h5"

# --- Processing Parameters ---
STITCHING_WINDOW_TYPE = 'blackmanharris'  # or hann
EXPECTED_RMS = 1.39e-02 # Target RMS for scaling chunks and final signal

# --- VVVVV Final Reconstruction Sample Rate VVVVV ---
# This is the target sample rate for the final stitched signal and for evaluation
# Should be >= Nyquist for the total signal bandwidth (e.g., > 800 MHz for 400 MHz BW)
# Using 1.0 GHz as a reasonable value, much lower than the old 3.6 GHz GT rate.
FS_RECON_FINAL = 3613200000.0 # Hz (e.g., 1.0 GHz)
# --- ^^^^^ Final Reconstruction Sample Rate ^^^^^ ---

# --- Upsampling Parameters (for initial plot only) ---
PLOT_FIRST_UPSAMPLED = False # Plot of first chunk after upsampling to FS_RECON_FINAL

# --- Phase Correction Parameters ---
EFFECTIVE_OVERLAP_FACTOR_CORR = 0.25 # Overlap factor used for pilot extraction & stitching
# Lag parameters are NOT used by pilot tone method, only by old correlation method
# CORRELATION_MAX_LAG_FRACTION = 0.1
# CORRELATION_MAX_LAG_ABS = 100

# --- Pilot Tone Config ---
EXPECTED_PILOT_IN_DATA = True # Must be True if INPUT_FILENAME contains pilot
PILOT_FFT_FACTOR = 4 # Zero-padding factor for FFT in pilot extraction

# --- WPD Parameters ---
APPLY_WPD_CORRECTION = True # Control WPD step after pilot correction
WPD_WAVELET = 'db4'
WPD_LEVEL = 4

# --- Stitching Parameters ---
# (No specific separate config needed, uses EFFECTIVE_OVERLAP_FACTOR_CORR)

# --- Evaluation & Visualization ---
PLOT_LENGTH = 5000 # Samples to use for time-domain plots
SPECTRUM_YLIM_BOTTOM = -140 # dB limit for spectrum plots (adjust as needed)
EVAL_MIN_RELIABLE_SAMPLES = 10 # Min samples for reliable metrics

# --- Adaptive Equalizer Parameters ---
# Set False to test the Pilot+WPD approach without subsequent equalization
APPLY_LMS_EQUALIZER = True
LMS_NUM_TAPS = 21 # (Not used if APPLY_LMS_EQUALIZER is False)
LMS_MU = 1e-7    # (Not used if APPLY_LMS_EQUALIZER is False)


# --- Debugging Flags ---
# SKIP_ADAPTIVE_FILTERING: This name might be misleading now.
# It originally referred to a potential pre-processing filter step.
# Let's keep it, but ensure it doesn't conflict with other steps.
SKIP_ADAPTIVE_FILTERING = True # Set based on whether any pre-filter was intended