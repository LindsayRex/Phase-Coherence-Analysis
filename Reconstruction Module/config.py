# config.py
"""Configuration parameters for the signal stitching pipeline."""
import os # Needed if using relative path construction
import logging
logger = logging.getLogger(__name__)

# Inside functions:
logger.debug("Very detailed message")
logger.info("General progress message")
logger.warning("Something unexpected but recoverable happened")
logger.error("An error occurred", exc_info=True) # Include traceback



# --- Input File ---
# Make sure this points to the HDF5 file generated WITH the pilot tone
INPUT_FILENAME = "simulated_chunks_25GHz_400MHzBW_qam16_sdr56MHz.h5"
# Example using relative path from config.py location:
# _MODULE_DIR = os.path.dirname(__file__)
# _DATA_DIR = os.path.abspath(os.path.join(_MODULE_DIR, '..', 'Simulated_data'))
# _FILENAME_ONLY = "simulated_chunks_25GHz_400MHzBW_qam16_sdr56MHz.h5" # Correct name!
# INPUT_FILENAME = os.path.join(_DATA_DIR, _FILENAME_ONLY)

# --- Processing Parameters ---
STITCHING_WINDOW_TYPE = 'blackmanharris'
EXPECTED_RMS = 1.39e-02 # Target RMS

# --- Upsampling Parameters ---
PLOT_FIRST_UPSAMPLED = False # Set to False to avoid plot popup during runs

# --- Phase Correction Parameters ---
# These might still be used by stitching if overlap calc needs them, keep for now
EFFECTIVE_OVERLAP_FACTOR_CORR = 0.25 # Overlap factor used for pilot extraction & stitching
# Lag parameters are NOT used by pilot tone method, but keep if referenced elsewhere
CORRELATION_MAX_LAG_FRACTION = 0.1
CORRELATION_MAX_LAG_ABS = 100

# --- VVVVV PILOT TONE CONFIG VVVVV ---
# Flag for main.py to check if the loaded INPUT_FILENAME should contain pilot data
EXPECTED_PILOT_IN_DATA = True # Set to True when using pilot data file
PILOT_FFT_FACTOR = 4 # Factor to multiply overlap length by for FFT resolution (e.g., 4 -> 4x zero padding)

# --- WPD Parameters ---
APPLY_WPD_CORRECTION = False # CONTROL WPD: Set True to run WPD *after* pilot correction
WPD_WAVELET = 'db4'
WPD_LEVEL = 4


# --- Stitching Parameters ---
# (No specific separate config needed, uses EFFECTIVE_OVERLAP_FACTOR_CORR)


# --- Evaluation & Visualization ---
PLOT_LENGTH = 5000
SPECTRUM_YLIM_BOTTOM = -120 # Adjust based on expected noise floor with pilot
EVAL_MIN_RELIABLE_SAMPLES = 10


# --- VVVVV REMOVED BOUNDARY CANCELLATION PARAMS VVVVV ---
# BOUNDARY_MU = 1e-4
# BOUNDARY_TAPS = 11
# BOUNDARY_ITER_PER_SAMPLE = 5
# --- ^^^^^ REMOVED BOUNDARY CANCELLATION PARAMS ^^^^^ ---


# --- Adaptive Equalizer Parameters ---
# --- VVVVV ENSURE THIS IS FALSE FOR PILOT TEST VVVVV ---
APPLY_LMS_EQUALIZER = False # Disable post-stitch equalizer when testing pilot/WPD correction
# --- ^^^^^ ENSURE THIS IS FALSE FOR PILOT TEST ^^^^^ ---
LMS_NUM_TAPS = 21            # (Not used if APPLY_LMS_EQUALIZER is False)
LMS_MU = 1e-6                # (Not used if APPLY_LMS_EQUALIZER is False)


# --- Debugging Flags ---
SKIP_ADAPTIVE_FILTERING = True # Keep this consistent with previous runs