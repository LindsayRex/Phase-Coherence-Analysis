# main.py

import numpy as np
import sys
import time

# --- CORRECTED IMPORTS ---
from . import config
from . import data_loader
from . import preprocessing
# from . import phase_correction # Can comment out if not used
from . import boundary_correction # <--- IMPORT boundary_correction
from . import stitching
from . import evaluation
from . import visualization
from . import utils
from . import equalizer # Keep import for post-stitch EQ (but disable in config)
# --- END CORRECTED IMPORTS ---


def main(): # <<--- Start of main function
    """Runs the entire signal stitching and evaluation pipeline."""
    start_time = time.time()

    # 1. Load Data
    loaded_chunks, loaded_metadata, global_attrs = data_loader.load_data(config.INPUT_FILENAME)
    if loaded_chunks is None: sys.exit("Exiting due to data loading error.")
    if not data_loader.validate_global_attrs(global_attrs): sys.exit("Exiting due to invalid global attributes.")

    # Extract necessary parameters
    sdr_rate = global_attrs.get('sdr_sample_rate_hz')
    recon_rate = global_attrs.get('ground_truth_sample_rate_hz')
    base_overlap_factor = global_attrs.get('overlap_factor', 0.1)
    tuning_delay = global_attrs.get('tuning_delay_s', 5e-6)

    # 2. Initial Amplitude Scaling
    scaled_chunks = preprocessing.scale_chunks(loaded_chunks, config.EXPECTED_RMS)
    if scaled_chunks is None: sys.exit("Exiting due to critical error during initial scaling.")
    current_chunks = scaled_chunks

    # 3. Skip Adaptive Filtering (Placeholder)
    if config.SKIP_ADAPTIVE_FILTERING: print("\n--- SKIPPING Adaptive Filtering Pre-Alignment ---")

    # 4. Upsampling
    upsampled_chunks = preprocessing.upsample_chunks(
        current_chunks, sdr_rate, recon_rate, plot_first=config.PLOT_FIRST_UPSAMPLED
    )
    # The output of upsampling is the input for boundary cancellation
    current_chunks = upsampled_chunks


    # --- VVVVV REPLACE Phase Correction with Boundary Cancellation VVVVV ---
    # 5. Adaptive Boundary Cancellation
    # Calculate effective overlap factor needed by boundary cancellation and later stitching
    effective_overlap = max(config.EFFECTIVE_OVERLAP_FACTOR_CORR, base_overlap_factor)

    corrected_boundary_chunks = boundary_correction.adaptive_boundary_cancellation(
        current_chunks, # Input the upsampled chunks
        recon_rate,
        effective_overlap, # Pass the overlap factor
        mu=config.BOUNDARY_MU, # Get params from config
        num_taps=config.BOUNDARY_TAPS,
        num_iterations_per_sample=config.BOUNDARY_ITER_PER_SAMPLE
    )
    # Check if boundary correction returned valid results
    if corrected_boundary_chunks is None or len(corrected_boundary_chunks) != len(upsampled_chunks):
        print("Error during boundary cancellation. Proceeding with uncorrected chunks.")
        current_chunks = upsampled_chunks # Fallback to use chunks before boundary correction
    else:
        current_chunks = corrected_boundary_chunks # Use the chunks processed by boundary cancellation
    # --- ^^^^^ END Boundary Cancellation Step ^^^^^ ---


    # 6. WPD Phase Correction (Should still be skipped)
    if config.APPLY_WPD_CORRECTION:
        print("\n--- WARNING: Skipping WPD Correction ---")
    else:
        print("\n--- SKIPPING Wavelet-Based Phase Correction ---")


    # 7. Pre-Stitching Normalization
    # Input is now the output of boundary cancellation (or upsampling if BC failed)
    normalized_chunks_stitch = stitching.normalize_chunks_pre_stitch(
        current_chunks,
        config.EXPECTED_RMS
    )

    # 8. Stitching
    chunk_duration_s = 0
    if len(loaded_chunks) > 0 and sdr_rate is not None and sdr_rate > 0: chunk_duration_s = len(loaded_chunks[0]) / sdr_rate
    elif len(normalized_chunks_stitch) > 0 and recon_rate is not None and recon_rate > 0: chunk_duration_s = len(normalized_chunks_stitch[0]) / recon_rate; print("Warning: Using processed chunk length for stitching duration estimate.")

    # Stitch using the chunks processed by boundary cancellation (and normalized)
    raw_stitched_signal, sum_windows = stitching.stitch_signal(
        normalized_chunks_stitch,
        sdr_rate, recon_rate, effective_overlap, tuning_delay, config.STITCHING_WINDOW_TYPE
    )
    if raw_stitched_signal is None: sys.exit("Exiting due to stitching error.")

    # 9. Post-Stitching Normalization
    final_stitched_signal = stitching.normalize_stitched_signal(
        raw_stitched_signal, sum_windows, config.EXPECTED_RMS
    )
    if final_stitched_signal is None: sys.exit("Exiting due to post-stitching normalization error.")


    # --- VVVVV ADAPTIVE EQUALIZATION STEP (Ensure DISABLED in config.py) VVVVV ---
    signal_for_evaluation = final_stitched_signal # Start with output of normalization

    if config.APPLY_LMS_EQUALIZER: # Check flag in config
        print("\n--- WARNING: Post-stitch equalizer is enabled, but boundary cancellation was applied. Consider disabling one. Running equalizer anyway... ---")
        qam16_points = np.array([ (r + 1j*i) / np.sqrt(10)
                                for r in [-3,-1,1,3] for i in [-3,-1,1,3] ])
        equalized_signal = equalizer.cma_equalizer(
            signal_for_evaluation, config.LMS_NUM_TAPS, config.LMS_MU, constellation=qam16_points
        )
        if equalized_signal is not None and len(equalized_signal)>0:
             rms_after_eq = np.sqrt(np.mean(np.abs(equalized_signal)**2))
             print(f"RMS after CMA Equalization: {rms_after_eq:.4e} (Target was {config.EXPECTED_RMS:.4e})")
             signal_for_evaluation = equalized_signal # Update signal if equalizer ran successfully
        else:
             print("Warning: Equalizer returned None or empty array. Using signal before equalization.")
    else:
        print("\n--- SKIPPING Post-Stitch CMA Equalization ---")
    # --- ^^^^^ END OF ADAPTIVE EQUALIZATION STEP ^^^^^ ---


    # 10. Evaluation
    total_recon_duration = 0.0
    if signal_for_evaluation is not None and recon_rate is not None and recon_rate > 0: total_recon_duration = len(signal_for_evaluation) / recon_rate
    gt_signal, t_vector = evaluation.regenerate_ground_truth(global_attrs, total_recon_duration, recon_rate, config.EXPECTED_RMS)
    metrics = {'mse': np.inf, 'nmse': np.inf, 'evm': np.inf}
    recon_signal_aligned_for_plot = signal_for_evaluation
    reliable_indices_eval = np.array([])
    if sum_windows is not None:
        max_len_reliable = len(signal_for_evaluation) if signal_for_evaluation is not None else 0
        reliable_indices_eval = np.where(sum_windows >= 1e-6)[0]
        if max_len_reliable > 0: reliable_indices_eval = reliable_indices_eval[reliable_indices_eval < max_len_reliable]
        else: reliable_indices_eval = np.array([])

    if gt_signal is not None and signal_for_evaluation is not None:
        metrics, recon_signal_aligned_for_plot = evaluation.calculate_metrics(gt_signal, signal_for_evaluation, reliable_indices_eval, config.EVAL_MIN_RELIABLE_SAMPLES)
    elif gt_signal is None: print("Skipping metrics calculation as Ground Truth generation failed.")
    else: print("Skipping metrics calculation as signal for evaluation is invalid.")

    # 11. Visualization
    if t_vector is not None and gt_signal is not None and recon_signal_aligned_for_plot is not None:
        visualization.plot_time_domain(t_vector, gt_signal, recon_signal_aligned_for_plot, config.EXPECTED_RMS, metrics['evm'], config.PLOT_LENGTH)
        visualization.plot_spectrum(gt_signal, recon_signal_aligned_for_plot, recon_rate, config.SPECTRUM_YLIM_BOTTOM)
    else: print("Skipping plots as Ground Truth, time vector, or reconstructed signal is invalid.")

    end_time = time.time()
    print(f"\nTotal script execution time: {end_time - start_time:.2f} seconds.")
    print("\nScript finished.")

# This part remains at the top level
if __name__ == "__main__":
    main()