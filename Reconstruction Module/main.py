# main.py

import numpy as np
import sys
import time
import logging
from scipy import signal as sig # Needed for resampling GT

# --- Setup Logging ---
from . import log_config
log_config.setup_logging(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

# --- Imports ---
from . import config
from . import data_loader
from . import preprocessing
from . import pilot_phase_correction
from . import spectral_stitching # <-- Using this for stitching
from . import evaluation
from . import visualization
from . import utils
from . import equalizer # <--- Need this for CMA
# --- End Imports ---


def main():
    """Runs the REVISED signal stitching pipeline using FREQUENCY DOMAIN STITCHING."""
    logger.info("="*50); logger.info(" Starting REVISED Reconstruction Pipeline Run ")
    logger.info(" Pipeline: Load -> Scale -> PILOT CORR -> FREQ DOMAIN STITCH -> EQUALIZE -> Eval ")
    logger.info("="*50); logger.info(f"Using config file source: {config.__file__}")
    start_time = time.time()

    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    plot_dir = f"run_plots_{run_timestamp}"

    # 1. Load Data
    logger.info("--- Step 1: Loading Data ---")
    loaded_sdr_chunks, loaded_metadata, global_attrs, gt_baseband_sim, gt_sim_rate = data_loader.load_data(config.INPUT_FILENAME)
    if loaded_sdr_chunks is None or gt_baseband_sim is None: logger.critical("Exiting: Data loading failed."); sys.exit(1)
    if not data_loader.validate_global_attrs(global_attrs): logger.critical("Exiting: Invalid global attributes."); sys.exit(1)
    if config.EXPECTED_PILOT_IN_DATA and not global_attrs.get('pilot_tone_added', False): logger.critical(f"Config expects pilot but metadata indicates none."); sys.exit("Regenerate data.")
    logger.info("Data loading successful.")

    # Extract parameters
    sdr_rate = global_attrs.get('sdr_sample_rate_hz'); final_recon_rate = config.FS_RECON_FINAL
    base_overlap_factor = global_attrs.get('overlap_factor', 0.1); tuning_delay = global_attrs.get('tuning_delay_s', 5e-6)
    logger.info(f"Extracted rates: SDR={sdr_rate/1e6:.1f} MHz, Target Final={final_recon_rate/1e6:.1f} MHz")

    # 2. Amplitude Scaling (SDR rate)
    logger.info(f"--- Step 2: Initial Amplitude Scaling (SDR Rate) ---")
    sdr_rate_chunks_scaled = preprocessing.scale_chunks(loaded_sdr_chunks, config.EXPECTED_RMS)
    if sdr_rate_chunks_scaled is None: logger.critical("Exiting: Scaling failed."); sys.exit(1)

    # 3. Pilot Phase Correction (SDR rate)
    logger.info("--- Step 3: Phase Correction (SDR Rate) ---")
    effective_overlap_pilot = max(config.EFFECTIVE_OVERLAP_FACTOR_CORR, base_overlap_factor)
    corrected_sdr_chunks = sdr_rate_chunks_scaled
    if config.EXPECTED_PILOT_IN_DATA:
        corrected_sdr_chunks, estimated_phases = pilot_phase_correction.correct_phase_pilot_tone(
            sdr_rate_chunks_scaled, loaded_metadata, global_attrs,
            apply_wpd=config.APPLY_WPD_CORRECTION, wavelet=config.WPD_WAVELET,
            wpd_level=config.WPD_LEVEL, n_fft_factor_pilot=config.PILOT_FFT_FACTOR
        )
        if corrected_sdr_chunks is None: logger.critical("Exiting: Pilot correction failed."); sys.exit(1)
    else: logger.info("Skipping pilot correction.")
    if config.APPLY_WPD_CORRECTION and config.EXPECTED_PILOT_IN_DATA: logger.info("WPD applied within pilot correction step.")
    elif config.APPLY_WPD_CORRECTION: logger.info("WPD applied independently (no pilot).")
    else: logger.info("WPD skipped.")


    # 4. Frequency Domain Stitching
    logger.info(f"--- Step 4: Frequency Domain Stitching (Output Rate: {final_recon_rate/1e6:.1f} MHz) ---")
    stitched_signal_unnormalized = spectral_stitching.frequency_domain_stitch(
        sdr_rate_chunks=corrected_sdr_chunks, metadata=loaded_metadata, global_attrs=global_attrs,
        fs_recon_final=final_recon_rate, freq_domain_window=config.STITCHING_WINDOW_TYPE,
        target_rms=None # Do final RMS scaling *after* equalizer if enabled
    )
    if stitched_signal_unnormalized is None: logger.critical("Exiting: Frequency domain stitching failed."); sys.exit(1)
    logger.info(f"Frequency domain stitching complete. Output len: {len(stitched_signal_unnormalized)}")

    # --- VVVVV STEP 5: Adaptive Equalizer (Optional) VVVVV ---
    logger.info("--- Step 5: Post-Stitch Adaptive Equalization ---")
    signal_after_eq = stitched_signal_unnormalized # Start with stitcher output
    if config.APPLY_LMS_EQUALIZER:
        if config.EXPECTED_PILOT_IN_DATA: logger.warning("Running Post-stitch equalizer after pilot tone correction.")
        # Define constellation for CMA R2 calculation
        qam16_points = np.array([ (r + 1j*i) / np.sqrt(10) for r in [-3,-1,1,3] for i in [-3,-1,1,3] ])
        signal_after_eq = equalizer.cma_equalizer(
            stitched_signal_unnormalized, # Input the signal BEFORE final RMS scaling
            config.LMS_NUM_TAPS, config.LMS_MU, constellation=qam16_points
        )
        if signal_after_eq is None or len(signal_after_eq)==0:
             logger.error("Equalizer failed or returned empty. Using signal before equalization.")
             signal_after_eq = stitched_signal_unnormalized # Fallback
        else:
             logger.info("CMA equalization applied.")
    else:
        logger.info("Skipping Post-Stitch CMA Equalization (as per config).")
    # --- ^^^^^ END STEP 5 ^^^^^ ---


    # --- VVVVV STEP 6: Final RMS Scaling VVVVV ---
    logger.info("--- Step 6: Final RMS Scaling ---")
    # Apply final scaling to the output of the equalizer (or stitcher if EQ skipped)
    rms_before_final_scale = np.sqrt(np.mean(np.abs(signal_after_eq)**2)) if len(signal_after_eq) > 0 else 0.0
    logger.info(f"RMS before final scaling: {rms_before_final_scale:.4e}")
    final_signal = signal_after_eq.copy() # Make a copy
    if rms_before_final_scale > 1e-15:
         final_scale_factor = config.EXPECTED_RMS / rms_before_final_scale
         final_signal *= final_scale_factor
         logger.info(f"Applied final scaling factor {final_scale_factor:.4f} to match target RMS ({config.EXPECTED_RMS:.4e}).")
         # Verify final RMS
         rms_final_check = np.sqrt(np.mean(np.abs(final_signal)**2))
         logger.info(f"Final RMS after scaling: {rms_final_check:.4e}")
    else:
         logger.warning("Signal RMS near zero before final scaling. Scaling skipped.")
         final_signal.fill(0) # Zero out if no power
    # --- ^^^^^ END STEP 6 ^^^^^ ---


    # --- VVVVV STEP 7: Evaluation VVVVV ---
    logger.info("--- Step 7: Evaluation ---")
    signal_for_evaluation = final_signal # Use the final scaled signal
    gt_signal_eval_rate = None; t_vector = None; metrics = {'mse': np.inf, 'nmse': np.inf, 'evm': np.inf}
    recon_signal_aligned_for_plot = signal_for_evaluation # Default

    if signal_for_evaluation is not None and final_recon_rate > 0:
        total_recon_duration = len(signal_for_evaluation) / final_recon_rate
        logger.info(f"Evaluating signal: Duration={total_recon_duration*1e6:.1f} us, Rate={final_recon_rate/1e6:.1f} MHz.")
        # Resample Ground Truth
        logger.info("Resampling loaded Ground Truth Baseband...")
        if gt_sim_rate == final_recon_rate: gt_signal_eval_rate = gt_baseband_sim
        else:
             try:
                  gt_sim_rate_int = int(np.round(gt_sim_rate)); final_recon_rate_int = int(np.round(final_recon_rate))
                  common = np.gcd(final_recon_rate_int, gt_sim_rate_int); up = final_recon_rate_int // common; down = gt_sim_rate_int // common
                  logger.debug(f"GT Resampling: Up={up}, Down={down}")
                  gt_real = sig.resample_poly(gt_baseband_sim.real.astype(np.float64), up, down)
                  gt_imag = sig.resample_poly(gt_baseband_sim.imag.astype(np.float64), up, down)
                  gt_signal_eval_rate = (gt_real + 1j * gt_imag).astype(np.complex128)
             except Exception as e: logger.error(f"Error resampling GT: {e}"); gt_signal_eval_rate = None
        # Adjust length and normalize resampled GT
        if gt_signal_eval_rate is not None:
             num_samples_eval = len(signal_for_evaluation); current_len = len(gt_signal_eval_rate)
             if current_len > num_samples_eval: gt_signal_eval_rate = gt_signal_eval_rate[:num_samples_eval]
             elif current_len < num_samples_eval: gt_signal_eval_rate = np.pad(gt_signal_eval_rate,(0,num_samples_eval-current_len))
             gt_rms = np.sqrt(np.mean(np.abs(gt_signal_eval_rate)**2))
             if gt_rms > 1e-15: gt_signal_eval_rate *= (config.EXPECTED_RMS / gt_rms)
             else: logger.warning("Resampled GT zero RMS."); gt_signal_eval_rate.fill(0)
             logger.info(f"Resampled GT adjusted: Len={len(gt_signal_eval_rate)}, RMS={config.EXPECTED_RMS:.2e}")
             t_vector = np.linspace(0, total_recon_duration, len(gt_signal_eval_rate), endpoint=False) if len(gt_signal_eval_rate)>0 else np.array([])
        else: logger.error("GT signal unavailable for evaluation.")
    else: logger.warning("Cannot evaluate due to invalid signal or rate.")

    # Calculate metrics
    if gt_signal_eval_rate is not None and signal_for_evaluation is not None:
        if len(gt_signal_eval_rate) != len(signal_for_evaluation): logger.error("Length mismatch GT vs Recon for metrics.")
        else:
             # Assuming all samples are reliable after freq domain stitching
             reliable_indices_eval = np.arange(len(signal_for_evaluation))
             logger.debug(f"Calculating metrics using {len(reliable_indices_eval)} indices.")
             metrics, recon_signal_aligned_for_plot = evaluation.calculate_metrics(gt_signal_eval_rate, signal_for_evaluation, reliable_indices_eval, config.EVAL_MIN_RELIABLE_SAMPLES)
    else: logger.warning("Skipping metrics calculation.")
    # --- ^^^^^ END STEP 7 ^^^^^ ---


    # --- VVVVV STEP 8: Visualization VVVVV ---
    logger.info("--- Step 8: Visualization ---")
    if t_vector is not None and gt_signal_eval_rate is not None and recon_signal_aligned_for_plot is not None:
        logger.info(f"Generating plots at final rate {final_recon_rate/1e6:.1f} MHz...")
        visualization.plot_time_domain(t_vector, gt_signal_eval_rate, recon_signal_aligned_for_plot, config.EXPECTED_RMS, metrics['evm'], config.PLOT_LENGTH, filename_suffix=f"run_{run_timestamp}", plot_dir=plot_dir)
        visualization.plot_spectrum(gt_signal_eval_rate, recon_signal_aligned_for_plot, final_recon_rate, config.SPECTRUM_YLIM_BOTTOM, filename_suffix=f"run_{run_timestamp}", plot_dir=plot_dir)
        visualization.plot_constellation(gt_signal_eval_rate, title="Ground Truth Constellation (Resampled)", filename_suffix=f"gt_resampled_{run_timestamp}", plot_dir=plot_dir)
        visualization.plot_constellation(signal_for_evaluation, title=f"Reconstructed Constellation (Before Align)", filename_suffix=f"recon_final_{run_timestamp}", plot_dir=plot_dir)
        visualization.plot_constellation(recon_signal_aligned_for_plot, title=f"Reconstructed Constellation (Plot Aligned, EVM={metrics['evm']:.2f}%)", filename_suffix=f"recon_aligned_{run_timestamp}", plot_dir=plot_dir)
        logger.info(f"Plots saved to directory: {plot_dir}")
    else: logger.warning("Skipping plots generation.")
    # --- ^^^^^ END STEP 8 ^^^^^ ---


    end_time = time.time()
    logger.info(f"Total script execution time: {end_time - start_time:.2f} seconds.")
    logger.info("--- Reconstruction Pipeline Finished ---")
    logger.info("="*50)


if __name__ == "__main__":
    main()