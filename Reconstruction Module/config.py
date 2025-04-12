# config.py
"""Configuration parameters for the signal stitching pipeline."""
import os # Needed if using relative path construction

# --- VVVVV REMOVED LOGGING SETUP FROM CONFIG VVVVV ---
# import logging
# logger = logging.getLogger(__name__)
# logger.debug(...) etc. - REMOVE ALL LOGGER CALLS
# --- ^^^^^ REMOVED LOGGING SETUP FROM CONFIG ^^^^^ ---


# --- Input File ---
# Make sure this points to the HDF5 file generated WITH the pilot tone
INPUT_FILENAME = "simulated_chunks_25GHz_400MHzBW_qam16_sdr56MHz.h5"
# Example using relative path from config.py location:
# Assumes Reconstruction Module is one level down from the project root
# and Simulated_data is also one level down from the project root.
# _SCRIPT_DIR = os.path.dirname(__file__)
# _PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
# _DATA_DIR = os.path.join(_PROJECT_ROOT, 'Simulated_data')
# _FILENAME_ONLY = "simulated_chunks_25GHz_400MHzBW_qam16_sdr56MHz.h5" # Ensure this exists
# if os.path.exists(os.path.join(_DATA_DIR, _FILENAME_ONLY)):
#      INPUT_FILENAME = os.path.join(_DATA_DIR, _FILENAME_ONLY)
# else:
#      # Fallback or error if file not found in expected relative location
#      print(f"WARNING: Could not find data file at relative path: {os.path.join(_DATA_DIR, _FILENAME_ONLY)}")
#      # Keep the original INPUT_FILENAME or raise an error
#      pass


# --- Processing Parameters ---
STITCHING_WINDOW_TYPE = 'blackmanharris'
EXPECTED_RMS = 1.39e-02 # Target RMS

# --- Upsampling Parameters ---
PLOT_FIRST_UPSAMPLED = False # Set to False to avoid plot popup during runs

# --- Phase Correction Parameters ---
EFFECTIVE_OVERLAP_FACTOR_CORR = 0.25 # Overlap factor used for pilot extraction & stitching
# Lag parameters are NOT used by pilot tone method
# CORRELATION_MAX_LAG_FRACTION = 0.1 # Can be commented out
# CORRELATION_MAX_LAG_ABS = 100      # Can be commented out

# --- Pilot Tone Config ---
EXPECTED_PILOT_IN_DATA = True
PILOT_FFT_FACTOR = 4 # Factor for FFT zero-padding (e.g., 4)

# --- WPD Parameters ---
APPLY_WPD_CORRECTION = True # CONTROL WPD: Set True/False as needed
WPD_WAVELET = 'db4'
WPD_LEVEL = 4


# --- Stitching Parameters ---
# (No specific separate config needed)


# --- Evaluation & Visualization ---
PLOT_LENGTH = 5000
SPECTRUM_YLIM_BOTTOM = -120
EVAL_MIN_RELIABLE_SAMPLES = 10


# --- Adaptive Equalizer Parameters ---
# Set True/False depending on whether you want to run the post-stitch equalizer
APPLY_LMS_EQUALIZER = True # Set to False when testing pilot/WPD alone
LMS_NUM_TAPS = 21
LMS_MU = 1e-7 # Start very small if APPLY_LMS_EQUALIZER is True


# --- Debugging Flags ---
SKIP_ADAPTIVE_FILTERING = True # Keep False if you want to simulate it