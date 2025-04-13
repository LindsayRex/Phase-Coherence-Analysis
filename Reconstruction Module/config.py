# config.py
"""Configuration parameters for the signal stitching pipeline."""
import os
import numpy as np

# --- Input File ---
# Make sure this points to the HDF5 file generated WITH the pilot tone
# AND containing the ground_truth_baseband dataset.
# Default relative path assumes running from the package root or main.py's directory
# Adjust if running from a different location.
INPUT_FILENAME = os.path.join("..", "Simulated_data", "simulated_chunks_25GHz_400MHzBW_qam16_sdr56MHz.h5")


# --- Processing Parameters ---
EXPECTED_RMS = 1.39e-02 # Target RMS for scaling chunks and final signal

# --- Final Reconstruction Sample Rate ---
# This is the target sample rate for the final stitched signal and for evaluation
# Should be >= Nyquist for the total signal bandwidth (e.g., > 800 MHz for 400 MHz BW)
# Using 1.0 GHz as a reasonable value, lower than the old 3.6 GHz GT rate.
# Set to None to use the ground truth simulation rate (if available)
FS_RECON_FINAL = 3613200000.0 # Hz (e.g., 1.0 GHz)


# --- Phase Correction Configuration ---
EXPECTED_PILOT_IN_DATA = True # Set True if HDF5 contains pilot tone

# Option 1: Wavelet-based phase alignment (NEW)
APPLY_WAVELET_PHASE_ALIGNMENT = True # Enable the new WPD alignment method
WPD_WAVELET_ALIGN = 'db8' # Wavelet for alignment (e.g., 'db4', 'db8', 'sym8')
WPD_LEVEL_ALIGN = 4        # WPD level for alignment phase extraction

# Option 2: Original Pilot Tone FFT method (If APPLY_WAVELET_PHASE_ALIGNMENT is False)
PILOT_FFT_FACTOR = 4 # Zero-padding factor for FFT in pilot extraction (used by old method)

# Post-correction Denoising/Detrending (Independent)
APPLY_WPD_DENOISING = False # Enable/disable the *old* WPD detrending step (likely redundant now)
WPD_WAVELET = 'db4'      # Wavelet for the *old* WPD detrending
WPD_LEVEL = 4            # Level for the *old* WPD detrending

# Pilot Tone Removal (New Step - applies *after* phase alignment)
APPLY_PILOT_REMOVAL = True # Enable/disable notch filter for pilot tone
PILOT_NOTCH_Q = 30         # Quality factor for the notch filter


# --- Stitching Parameters ---
# Overlap factor used *during* stitching (can be different from capture overlap)
EFFECTIVE_OVERLAP_FACTOR_CORR = 0.25 # Overlap factor used for combining windowed FFTs
# Window function for spectral stitching (applied in time domain before FFT)
STITCHING_WINDOW_TYPE = 'hann' # Recommended: 'hann'. Options: 'blackmanharris', 'rect' (None)
# STITCHING_OVERLAP_FACTOR = 0.25 # Optional: If changing, requires spectral_stitching.py logic change


# --- Adaptive Equalizer Configuration ---
APPLY_ADAPTIVE_EQUALIZER = True # Master switch for equalization step
EQUALIZER_TYPE = 'MMA'          # Algorithm type: 'CMA' or 'MMA'
EQ_NUM_TAPS = 21                # Number of equalizer taps (must be odd)
EQ_MU = 5e-6                    # Step size (learning rate) - start small for MMA/stability


# --- Evaluation & Visualization ---
PLOT_LENGTH = 5000 # Samples to use for time-domain plots
SPECTRUM_YLIM_BOTTOM = -140 # dB limit for spectrum plots (adjust as needed)
EVAL_MIN_RELIABLE_SAMPLES = 10 # Min samples for reliable metrics

# --- Debugging Flags ---
# SKIP_ADAPTIVE_FILTERING: Legacy flag, likely not needed. Set to True.
SKIP_ADAPTIVE_FILTERING = True