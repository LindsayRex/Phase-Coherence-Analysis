# config.py
"""Configuration parameters for the signal stitching pipeline."""

# --- Input File ---
INPUT_FILENAME = "simulated_chunks_25GHz_400MHzBW_qam16_sdr56MHz.h5"

# --- Processing Parameters ---
STITCHING_WINDOW_TYPE = 'blackmanharris'
EXPECTED_RMS = 1.39e-02 # Target RMS after initial scaling and for pre/post stitching normalization

# --- Upsampling Parameters ---
# (Specific parameters like filter choice are often kept within the function)
PLOT_FIRST_UPSAMPLED = True # Flag to plot the first chunk after upsampling

# --- Phase Correction Parameters ---
EFFECTIVE_OVERLAP_FACTOR_CORR = 0.25 # Minimum overlap used for phase correlation step
CORRELATION_MAX_LAG_FRACTION = 0.1 # Fraction of overlap samples to search for time lag
CORRELATION_MAX_LAG_ABS = 100     # Absolute maximum lag samples to search

# --- WPD Parameters ---
WPD_WAVELET = 'db4'
WPD_LEVEL = 4
# NOTE: WPD implementation is currently for intra-chunk detrending and had issues.
APPLY_WPD_CORRECTION = False # Set to False to easily skip the WPD step

# --- Stitching Parameters ---
# Stitching uses the effective overlap calculated based on CORR factor for consistency
# STITCHING_OVERLAP_FACTOR is derived from EFFECTIVE_OVERLAP_FACTOR_CORR in main script

# --- Evaluation & Visualization ---
PLOT_LENGTH = 5000      # Number of samples for time-domain plots
SPECTRUM_YLIM_BOTTOM = -100 # Bottom Y-limit for spectrum plot (dB)
EVAL_MIN_RELIABLE_SAMPLES = 10 # Minimum samples needed for metrics

# --- Boundary Cancellation Parame  ters ---
BOUNDARY_MU = 1e-4           # Step size for boundary LMS (tune this!)
BOUNDARY_TAPS = 11           # Filter taps for boundary cancellation (odd)
BOUNDARY_ITER_PER_SAMPLE = 5 # Number of LMS updates per overlap sample

# --- Adaptive Equalizer Parameters ---
APPLY_LMS_EQUALIZER = False   # Set to True to enable, False to disable
LMS_NUM_TAPS = 21            # Number of equalizer taps (e.g., 11, 21, 31 - MUST BE ODD)
LMS_MU = 1e-6                # LMS/CMA step size (LEARNING RATE) - START VERY SMALL for CMA (e.g., 1e-6, 1e-7)

# --- Debugging Flags ---
SKIP_ADAPTIVE_FILTERING = False # As per the original script's logic