# main.py

import numpy as np
import sys
import time
import logging

# --- Setup Logging (Call setup function from log_config) ---
from . import log_config
# Configure logging level here ONCE for the entire application
# --- VVVVV CORRECTED LEVEL VVVVV ---
log_config.setup_logging(level=logging.DEBUG, log_dir="run_logs")

# --- End Logging Setup ---

# --- Imports for Modules ---
from . import config
from . import data_loader
from . import preprocessing
from . import pilot_phase_correction # Using pilot tone approach
from . import stitching
from . import evaluation
from . import visualization
from . import utils
from . import equalizer # Keep import, although likely disabled in config
# --- End Imports ---

# Get logger for this main module
logger = logging.getLogger(__name__)

def main():
    """Runs the entire signal stitching and evaluation pipeline."""
    logger.info("="*50)
    logger.info(" Starting Reconstruction Pipeline Run ")
    logger.info("="*50)
    logger.info(f"Using config file source: {config.__file__}")
    start_time = time.time()

    # --- Create unique identifier for this run (for plot filenames) ---
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    plot_dir = f"run_plots_{run_timestamp}" # Directory to save plots for this run


    # 1. Load Data
    logger.info("--- Step 1: Loading Data ---")
    logger.info(f"Attempting to load data from: {config.INPUT_FILENAME}")
    loaded_chunks, loaded_metadata, global_attrs = data_loader.load_data(config.INPUT_FILENAME)

    if loaded_chunks is None or not loaded_chunks:
        logger.critical("Exiting: Data loading failed or returned no chunks.")
        sys.exit(1)
    if not data_loader.validate_global_attrs(global_attrs):
        logger.critical("Exiting: Invalid global attributes found in data file.")
        sys.exit(1)

    # Check for pilot tone consistency between config and loaded data
    if config.EXPECTED_PILOT_IN_DATA and not global_attrs.get('pilot_tone_added', False):
         logger.critical(f"Config expects pilot (EXPECTED_PILOT_IN_DATA=True) but loaded file metadata indicates none.")
         sys.exit("Regenerate data with pilot or set EXPECTED_PILOT_IN_DATA=False in config.")
    elif not config.EXPECTED_PILOT_IN_DATA and global_attrs.get('pilot_tone_added', True):
         logger.warning(f"Config expects NO pilot (EXPECTED_PILOT_IN_DATA=False) but loaded file contains pilot metadata. Will skip pilot correction.")
    elif config.EXPECTED_PILOT_IN_DATA:
         logger.info("Pilot tone expected and metadata confirms presence.")
    else: # Not expecting pilot, and file doesn't have it (or flag missing)
         logger.info("Proceeding without pilot tone correction as per config.")

    # Extract parameters safely using .get() with defaults
    sdr_rate = global_attrs.get('sdr_sample_rate_hz')
    recon_rate = global_attrs.get('ground_truth_sample_rate_hz')
    base_overlap_factor = global_attrs.get('overlap_factor', 0.1)
    tuning_delay = global_attrs.get('tuning_delay_s', 5e-6)
    logger.info(f"Extracted parameters: SDR Rate={sdr_rate/1e6} MHz, Recon Rate={recon_rate/1e6} MHz, Overlap={base_overlap_factor}, Tuning Delay={tuning_delay*1e6} us")


    # 2. Initial Amplitude Scaling
    logger.info("--- Step 2: Initial Amplitude Scaling ---")
    scaled_chunks = preprocessing.scale_chunks(loaded_chunks, config.EXPECTED_RMS)
    if scaled_chunks is None:
        logger.critical("Exiting: Critical error during initial scaling.")
        sys.exit(1)
    current_chunks = scaled_chunks


    # 3. Skip Adaptive Filtering (Placeholder)
    if config.SKIP_ADAPTIVE_FILTERING:
        logger.info("--- Step 3: Skipping Adaptive Filtering Pre-Alignment (Config Flag) ---")


    # 4. Upsampling
    logger.info("--- Step 4: Upsampling Chunks ---")
    upsampled_chunks = preprocessing.upsample_chunks(
        current_chunks, sdr_rate, recon_rate, plot_first=config.PLOT_FIRST_UPSAMPLED
    )
    current_chunks = upsampled_chunks


    # 5. Phase Correction (Pilot Tone + Optional WPD)
    logger.info("--- Step 5: Phase Correction ---")
    effective_overlap = max(config.EFFECTIVE_OVERLAP_FACTOR_CORR, base_overlap_factor) # Used for stitching too
    logger.info(f"Using effective overlap factor: {effective_overlap:.3f}")

    if config.EXPECTED_PILOT_IN_DATA:
        logger.info("Calling pilot tone phase correction...")
        corrected_chunks, estimated_phases = pilot_phase_correction.correct_phase_pilot_tone(
            current_chunks, # Pass upsampled chunks now
            loaded_metadata,
            global_attrs,
            apply_wpd=config.APPLY_WPD_CORRECTION,
            wavelet=config.WPD_WAVELET,
            wpd_level=config.WPD_LEVEL,
            n_fft_factor_pilot=config.PILOT_FFT_FACTOR
        )
        if corrected_chunks is None or len(corrected_chunks) != len(upsampled_chunks):
            logger.error("Pilot/WPD phase correction failed. Proceeding with uncorrected upsampled chunks.")
            current_chunks = upsampled_chunks # Fallback
        else:
            logger.info("Pilot tone phase correction (and optional WPD) applied.")
            current_chunks = corrected_chunks
            # Log estimated phases for analysis
            logger.debug("Estimated Cumulative Phases (radians):")
            for i, ph in enumerate(estimated_phases): logger.debug(f"  Chunk {i}: {ph:.4f}")
    else:
         logger.info("Skipping Pilot Tone Phase Correction (as per config)")
         if config.APPLY_WPD_CORRECTION:
              logger.info("Applying WPD correction independently to upsampled chunks...")
              # Call WPD function directly (it's defined in pilot_phase_correction module)
              wpd_corrected_chunks = pilot_phase_correction.correct_phase_wpd(
                   current_chunks, wavelet=config.WPD_WAVELET, level=config.WPD_LEVEL
              )
              current_chunks = wpd_corrected_chunks
         else: logger.info("Skipping WPD Correction (as per config)")


    # 6. WPD Step (Now integrated into step 5)
    logger.info("--- Step 6: WPD (Integrated within Step 5 if enabled) ---")


    # 7. Pre-Stitching Normalization
    logger.info("--- Step 7: Pre-Stitching Normalization ---")
    normalized_chunks_stitch = stitching.normalize_chunks_pre_stitch(
        current_chunks, config.EXPECTED_RMS
    )


    # 8. Stitching
    logger.info("--- Step 8: Stitching Signal ---")
    chunk_duration_s = 0
    if len(loaded_chunks) > 0 and sdr_rate > 0: chunk_duration_s = len(loaded_chunks[0]) / sdr_rate
    elif len(normalized_chunks_stitch) > 0 and recon_rate > 0: chunk_duration_s = len(normalized_chunks_stitch[0]) / recon_rate; logger.warning("Using processed chunk length for stitching duration estimate.")
    if chunk_duration_s <= 0: logger.warning("Could not determine chunk duration for stitching.")

    raw_stitched_signal, sum_windows = stitching.stitch_signal(
        normalized_chunks_stitch, sdr_rate, recon_rate, effective_overlap, tuning_delay, config.STITCHING_WINDOW_TYPE
    )
    if raw_stitched_signal is None: logger.critical("Exiting due to stitching error."); sys.exit(1)


    # 9. Post-Stitching Normalization
    logger.info("--- Step 9: Post-Stitching Normalization ---")
    final_stitched_signal = stitching.normalize_stitched_signal(
        raw_stitched_signal, sum_windows, config.EXPECTED_RMS
    )
    if final_stitched_signal is None: logger.critical("Exiting due to post-stitching normalization error."); sys.exit(1)


    # 10. Adaptive Equalizer (Check if enabled)
    logger.info("--- Step 10: Post-Stitch Adaptive Equalization ---")
    signal_for_evaluation = final_stitched_signal # Default to signal before EQ
    if config.APPLY_LMS_EQUALIZER:
        if config.EXPECTED_PILOT_IN_DATA:
             logger.warning("Running Post-stitch equalizer even though pilot tone correction was applied. Check config.")
        # Define constellation for CMA R2 calculation (if needed)
        qam16_points = np.array([ (r + 1j*i) / np.sqrt(10) for r in [-3,-1,1,3] for i in [-3,-1,1,3] ])
        equalized_signal = equalizer.cma_equalizer(
            signal_for_evaluation, config.LMS_NUM_TAPS, config.LMS_MU, constellation=qam16_points
        )
        if equalized_signal is not None and len(equalized_signal)>0:
             rms_after_eq = np.sqrt(np.mean(np.abs(equalized_signal)**2))
             logger.info(f"RMS after CMA Equalization: {rms_after_eq:.4e} (Target was {config.EXPECTED_RMS:.4e})")
             signal_for_evaluation = equalized_signal # Use the equalized signal
        else:
             logger.warning("Equalizer failed or returned empty. Using signal before equalization.")
    else:
        logger.info("Skipping Post-Stitch CMA Equalization (as per config).")


    # 11. Evaluation
    logger.info("--- Step 11: Evaluation ---")
    total_recon_duration = 0.0; gt_signal=None; t_vector=None
    if signal_for_evaluation is not None and recon_rate > 0: total_recon_duration = len(signal_for_evaluation) / recon_rate
    if total_recon_duration > 0: gt_signal, t_vector = evaluation.regenerate_ground_truth(global_attrs, total_recon_duration, recon_rate, config.EXPECTED_RMS)
    metrics = {'mse': np.inf, 'nmse': np.inf, 'evm': np.inf}; recon_signal_aligned_for_plot = signal_for_evaluation
    reliable_indices_eval = np.array([]); max_len_reliable=0
    if sum_windows is not None: max_len_reliable = len(signal_for_evaluation) if signal_for_evaluation is not None else 0
    if sum_windows is not None and max_len_reliable > 0: reliable_indices_eval = np.where(sum_windows >= 1e-6)[0]; reliable_indices_eval = reliable_indices_eval[reliable_indices_eval < max_len_reliable]
    if gt_signal is not None and signal_for_evaluation is not None:
        metrics, recon_signal_aligned_for_plot = evaluation.calculate_metrics(gt_signal, signal_for_evaluation, reliable_indices_eval, config.EVAL_MIN_RELIABLE_SAMPLES)
    else: logger.warning("Skipping metrics calculation due to invalid GT or reconstructed signal.")


    # 12. Visualization
    logger.info("--- Step 12: Visualization ---")
    if t_vector is not None and gt_signal is not None and recon_signal_aligned_for_plot is not None:
        logger.info("Generating plots...")
        # Pass the unique run timestamp to group plots
        visualization.plot_time_domain(
            t_vector, gt_signal, recon_signal_aligned_for_plot,
            config.EXPECTED_RMS, metrics['evm'], config.PLOT_LENGTH,
            filename_suffix=f"run_{run_timestamp}", plot_dir=plot_dir
        )
        visualization.plot_spectrum(
            gt_signal, recon_signal_aligned_for_plot,
            recon_rate, config.SPECTRUM_YLIM_BOTTOM,
            filename_suffix=f"run_{run_timestamp}", plot_dir=plot_dir
        )
        visualization.plot_constellation(
            gt_signal, title="Ground Truth Constellation",
            filename_suffix=f"gt_{run_timestamp}", plot_dir=plot_dir
        )
        # Plot constellation *before* alignment might be useful too
        visualization.plot_constellation(
             signal_for_evaluation, title=f"Reconstructed Constellation (Final Eval Input)",
             filename_suffix=f"recon_eval_input_{run_timestamp}", plot_dir=plot_dir
        )
        visualization.plot_constellation(
            recon_signal_aligned_for_plot, title=f"Reconstructed Constellation (Plot Aligned, EVM={metrics['evm']:.2f}%)",
            filename_suffix=f"recon_aligned_{run_timestamp}", plot_dir=plot_dir
        )
        logger.info(f"Plots saved to directory: {plot_dir}")
    else:
        logger.warning("Skipping plots generation due to invalid input signals or time vector.")


    end_time = time.time()
    logger.info(f"Total script execution time: {end_time - start_time:.2f} seconds.")
    logger.info("--- Reconstruction Pipeline Finished ---")
    logger.info("="*50)

# Configure logging when the script is imported or run
# Moved setup call outside main to ensure logger is ready
log_config.setup_logging(level=logging.DEBUG, log_dir="run_logs") # Default to INFO

if __name__ == "__main__":
    # Optionally override log level from command line?
    # if len(sys.argv) > 1 and sys.argv[1].upper() == 'DEBUG':
    #     log_config.setup_logging(level=logging.DEBUG)
    main()