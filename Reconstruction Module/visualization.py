# visualization.py
import numpy as np
import matplotlib

# Set backend *before* importing pyplot
# 'Agg' is good for non-interactive saving to file
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
from . import utils
import logging
from . import log_config

# Setup logging for this module
log_config.setup_logging(level=logging.DEBUG, log_dir="run_logs")
logger = logging.getLogger(__name__)

def save_plot(fig, plot_filename, plot_dir="plots"): # Keep plot_dir here
    """Saves the current figure to a file."""
    try:
        os.makedirs(plot_dir, exist_ok=True) # Use the passed plot_dir
        full_path = os.path.join(plot_dir, plot_filename)
        fig.savefig(full_path, bbox_inches='tight', dpi=150)
        logger.info(f"Plot saved to: {full_path}")
        plt.close(fig) # Close the figure to free memory
    except Exception as e:
        logger.error(f"Failed to save plot {plot_filename}: {e}", exc_info=False)
        if 'fig' in locals() and fig is not None: plt.close(fig) # Ensure close on error


# --- VVVVV Add plot_dir argument VVVVV ---
def plot_time_domain(t_vector, gt_signal, recon_signal, target_rms, evm, plot_length,
                     filename_suffix="", plot_dir="plots"): # Added plot_dir
# --- ^^^^^ Add plot_dir argument ^^^^^ ---
    """Plots Time Domain signals and saves to file."""
    logger.info("Generating Time Domain Plot...")
    # Use a unique filename including the suffix
    plot_filename = f"plot_time_domain_{filename_suffix}_{time.strftime('%Y%m%d_%H%M%S')}.png"
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    try:
        # ... (rest of the plotting logic remains exactly the same) ...
        plot_samples = min(plot_length, len(t_vector), len(gt_signal), len(recon_signal))
        if plot_samples > 0:
            time_axis_plot = t_vector[:plot_samples] * 1e6
            # Plot GT, Recon, Error...
            # ... (plotting code using axs[0], axs[1], axs[2]) ...
            plt.tight_layout()
            save_plot(fig, plot_filename, plot_dir=plot_dir) # Pass plot_dir to helper
        else:
             logger.warning("Skipping time domain plot saving: Not enough samples.")
             plt.close(fig) # Close empty figure
    except Exception as e:
        logger.error(f"Error generating time domain plot: {e}", exc_info=True)
        if 'fig' in locals() and fig is not None: plt.close(fig)


# --- VVVVV Add plot_dir argument VVVVV ---
def plot_spectrum(gt_signal, recon_signal, recon_rate, spectrum_ylim_bottom,
                  filename_suffix="", plot_dir="plots"): # Added plot_dir
# --- ^^^^^ Add plot_dir argument ^^^^^ ---
    """Plots the power spectrum comparison."""
    logger.info("Generating Spectrum Plot...")
    plot_filename = f"plot_spectrum_{filename_suffix}_{time.strftime('%Y%m%d_%H%M%S')}.png"
    fig = plt.figure(figsize=(12, 7))
    plot_spectrum_flag = False

    try:
        # ... (rest of the plotting logic remains exactly the same) ...
        # Plot GT spectrum
        if gt_signal is not None and len(gt_signal) > 1 and recon_rate is not None:
            f_gt, spec_gt_db = utils.compute_spectrum(gt_signal, recon_rate)
            if len(f_gt) > 0: plt.plot(f_gt / 1e6, spec_gt_db, label='GT Spectrum', alpha=0.8); plot_spectrum_flag = True
        # Plot Recon spectrum
        if recon_signal is not None and len(recon_signal) > 1 and recon_rate is not None:
            f_recon, spec_recon_db = utils.compute_spectrum(recon_signal, recon_rate)
            if len(f_recon) > 0: plt.plot(f_recon / 1e6, spec_recon_db, label='Recon Spectrum (Plot Aligned)', ls='--', alpha=0.9); plot_spectrum_flag = True

        if plot_spectrum_flag:
            # ... (setting title, labels, ylims) ...
            plt.tight_layout()
            save_plot(fig, plot_filename, plot_dir=plot_dir) # Pass plot_dir to helper
        else:
            logger.warning("Skipping spectrum plot saving: No valid spectra computed.")
            plt.close(fig)
    except Exception as e:
        logger.error(f"Error generating spectrum plot: {e}", exc_info=True)
        if 'fig' in locals() and fig is not None: plt.close(fig)


# --- VVVVV Add plot_dir argument VVVVV ---
def plot_constellation(signal, title="Constellation Plot", filename_suffix="",
                       plot_dir="plots", num_points=5000): # Added plot_dir
# --- ^^^^^ Add plot_dir argument ^^^^^ ---
    """Plots Constellation Diagram and saves to file."""
    logger.info(f"Generating Constellation Plot: {title}")
    plot_filename = f"plot_constellation_{filename_suffix}_{time.strftime('%Y%m%d_%H%M%S')}.png"
    fig = plt.figure(figsize=(8, 8))

    try:
        # ... (rest of the plotting logic remains exactly the same) ...
        if signal is None or len(signal) == 0: logger.warning(f"Skipping constellation plot '{title}'"); plt.close(fig); return
        if len(signal) > num_points: step = len(signal) // num_points; plot_data = signal[::step]
        else: plot_data = signal
        plot_data_finite = plot_data[np.isfinite(plot_data)]
        if len(plot_data_finite) == 0: logger.warning(f"Skipping constellation plot '{title}': No finite points."); plt.close(fig); return

        plt.scatter(np.real(plot_data_finite), np.imag(plot_data_finite), s=5, alpha=0.5)
        # ... (setting labels, title, grid, limits) ...
        plt.tight_layout()
        save_plot(fig, plot_filename, plot_dir=plot_dir) # Pass plot_dir to helper
    except Exception as e:
        logger.error(f"Error generating constellation plot '{title}': {e}", exc_info=True)
        if 'fig' in locals() and fig is not None: plt.close(fig)