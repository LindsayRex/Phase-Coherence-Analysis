# visualization.py
import numpy as np
import matplotlib
# Set backend *before* importing pyplot for saving without display
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
from . import utils # Assuming utils.py is in the same package directory
import logging

# Get logger for this module
logger = logging.getLogger(__name__)
# DO NOT call log_config.setup_logging() here


def save_plot(fig, plot_filename, plot_dir="plots"):
    """Saves the current figure to a file."""
    try:
        if fig is None:
             logger.error(f"Cannot save plot '{plot_filename}', figure object is None.")
             return
        os.makedirs(plot_dir, exist_ok=True)
        full_path = os.path.join(plot_dir, plot_filename)
        fig.savefig(full_path, bbox_inches='tight', dpi=150)
        logger.info(f"Plot saved to: {full_path}")
    except Exception as e:
        logger.error(f"Failed to save plot {plot_filename}: {e}", exc_info=False)
    finally:
        # Ensure figure is closed even if saving fails or was skipped
        if fig is not None and plt.fignum_exists(fig.number):
             plt.close(fig)
             logger.debug(f"Closed plot figure {fig.number} for {plot_filename}")


def plot_time_domain(t_vector, gt_signal, recon_signal, target_rms, evm, plot_length,
                     filename_suffix="", plot_dir="plots"):
    """Plots Time Domain signals and saves to file."""
    logger.info("Generating Time Domain Plot...")
    plot_filename = f"plot_time_domain_{filename_suffix}_{time.strftime('%Y%m%d_%H%M%S')}.png"
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.number # Access figure number to ensure it's created before try block potentially fails

    try:
        # Input validation
        if t_vector is None or gt_signal is None or recon_signal is None:
             logger.warning("Skipping time domain plot due to None input.")
             save_plot(fig, f"skipped_{plot_filename}", plot_dir=plot_dir) # Save empty plot with indicator
             return

        plot_samples = min(plot_length, len(t_vector), len(gt_signal), len(recon_signal))
        logger.debug(f"Plotting first {plot_samples} samples for time domain.")

        if plot_samples > 0:
            time_axis_plot = t_vector[:plot_samples] * 1e6 # Time in microseconds

            # --- Plot GT ---
            gt_plot_data = gt_signal[:plot_samples]
            axs[0].plot(time_axis_plot, np.real(gt_plot_data), label='GT (Real)', linewidth=1.0)
            axs[0].plot(time_axis_plot, np.imag(gt_plot_data), label='GT (Imag)', alpha=0.7, linewidth=1.0)
            axs[0].set_title(f'Ground Truth (Target RMS={target_rms:.2e})')
            axs[0].set_xlabel('Time (µs)') # Add X label to bottom plot later
            axs[0].set_ylabel('Amplitude')
            axs[0].legend(fontsize='small', loc='upper right')
            axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
            ylim_gt_abs = target_rms * 4; max_val = np.nanmax(np.abs(gt_plot_data)) if np.any(np.isfinite(gt_plot_data)) else 0
            if max_val > 1e-9: ylim_gt_abs = max(ylim_gt_abs, max_val)
            axs[0].set_ylim(-ylim_gt_abs * 1.2, ylim_gt_abs * 1.2)

            # --- Plot Recon (Aligned) ---
            recon_plot_data = recon_signal[:plot_samples]
            axs[1].plot(time_axis_plot, np.nan_to_num(np.real(recon_plot_data)), label='Recon (Real)', linewidth=1.0)
            axs[1].plot(time_axis_plot, np.nan_to_num(np.imag(recon_plot_data)), label='Recon (Imag)', alpha=0.7, linewidth=1.0)
            title_recon = f'Reconstructed Signal (Plot Aligned)'
            if np.isfinite(evm): title_recon += f' / Eval EVM: {evm:.2f}%'
            axs[1].set_title(title_recon)
            axs[1].set_ylabel('Amplitude')
            axs[1].legend(fontsize='small', loc='upper right')
            axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
            ylim_recon_abs = ylim_gt_abs; max_val = np.nanmax(np.abs(recon_plot_data)) if np.any(np.isfinite(recon_plot_data)) else 0
            if max_val > 1e-9: ylim_recon_abs = max(ylim_gt_abs, max_val)
            axs[1].set_ylim(-ylim_recon_abs * 1.2, ylim_recon_abs * 1.2)

            # --- Plot Error ---
            error_signal = gt_plot_data - recon_plot_data # Error relative to the plotted recon signal
            axs[2].plot(time_axis_plot, np.nan_to_num(np.real(error_signal)), label='Error (Real)', linewidth=1.0)
            axs[2].plot(time_axis_plot, np.nan_to_num(np.imag(error_signal)), label='Error (Imag)', alpha=0.7, linewidth=1.0)
            axs[2].set_title('Error Signal (GT - Plot Aligned Recon)')
            axs[2].set_xlabel('Time (µs)') # X label on the bottom plot
            axs[2].set_ylabel('Amplitude')
            axs[2].legend(fontsize='small', loc='upper right')
            axs[2].grid(True, which='both', linestyle='--', linewidth=0.5)
            ylim_err_abs = ylim_recon_abs * 0.5; max_val = np.nanmax(np.abs(error_signal)) if np.any(np.isfinite(error_signal)) else 0
            if max_val > 1e-9: ylim_err_abs = max(ylim_err_abs, max_val)
            axs[2].set_ylim(-ylim_err_abs * 1.2, ylim_err_abs * 1.2)

            fig.suptitle("Time Domain Signal Comparison", fontsize=16) # Add overall figure title
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
            save_plot(fig, plot_filename, plot_dir=plot_dir)
        else:
             logger.warning("Skipping time domain plot saving: Not enough samples.")
             plt.close(fig) # Close empty figure if not plotted
    except Exception as e:
        logger.error(f"Error generating time domain plot: {e}", exc_info=True)
        # Ensure figure is closed on error, checking if fig exists and is valid
        if 'fig' in locals() and fig is not None and plt.fignum_exists(fig.number):
             plt.close(fig)


def plot_spectrum(gt_signal, recon_signal, recon_rate, spectrum_ylim_bottom,
                  filename_suffix="", plot_dir="plots"):
    """Plots the power spectrum comparison and saves to file."""
    logger.info("Generating Spectrum Plot...")
    plot_filename = f"plot_spectrum_{filename_suffix}_{time.strftime('%Y%m%d_%H%M%S')}.png"
    fig = plt.figure(figsize=(12, 7)); plot_spectrum_flag = False
    fig.number # Access figure number

    try:
        # Plot GT spectrum
        if gt_signal is not None and len(gt_signal) > 1 and recon_rate is not None:
            logger.debug("Computing GT spectrum...")
            f_gt, spec_gt_db = utils.compute_spectrum(gt_signal, recon_rate)
            if len(f_gt) > 0: plt.plot(f_gt / 1e6, spec_gt_db, label='GT Spectrum', alpha=0.8); plot_spectrum_flag = True
            else: logger.warning("GT spectrum computation returned empty.")
        else: logger.warning("Skipping GT spectrum plot (invalid input).")

        # Plot Recon spectrum
        if recon_signal is not None and len(recon_signal) > 1 and recon_rate is not None:
            logger.debug("Computing Recon spectrum...")
            f_recon, spec_recon_db = utils.compute_spectrum(recon_signal, recon_rate)
            if len(f_recon) > 0: plt.plot(f_recon / 1e6, spec_recon_db, label='Recon Spectrum (Plot Aligned)', ls='--', alpha=0.9); plot_spectrum_flag = True
            else: logger.warning("Reconstructed spectrum computation returned empty.")
        else: logger.warning("Skipping Reconstructed spectrum plot (invalid input).")

        if plot_spectrum_flag:
            plt.title('Power Spectrum Comparison') # Title
            plt.xlabel('Frequency (MHz)')         # X Label
            plt.ylabel('Power Spectrum (dB)')      # Y Label
            plt.ylim(bottom=spectrum_ylim_bottom)
            # Dynamic top ylim
            max_spec_val = -np.inf; ax = plt.gca()
            for line in ax.get_lines():
                 ydata = line.get_ydata()
                 finite_y = ydata[np.isfinite(ydata)]
                 if len(finite_y)>0: max_spec_val = max(max_spec_val, np.max(finite_y))
            if np.isfinite(max_spec_val): plt.ylim(bottom=spectrum_ylim_bottom, top=min(max_spec_val + 10, 20))
            else: plt.ylim(bottom=spectrum_ylim_bottom, top=0)
            plt.legend() # Legend
            plt.grid(True, which='both', linestyle='--')
            plt.tight_layout()
            save_plot(fig, plot_filename, plot_dir=plot_dir)
        else:
            logger.warning("Skipping spectrum plot saving: No valid spectra computed.")
            plt.close(fig)
    except Exception as e:
        logger.error(f"Error generating spectrum plot: {e}", exc_info=True)
        if 'fig' in locals() and fig is not None and plt.fignum_exists(fig.number): plt.close(fig)


def plot_constellation(signal, title="Constellation Plot", filename_suffix="",
                       plot_dir="plots", num_points=5000):
    """Plots Constellation Diagram and saves to file."""
    logger.info(f"Generating Constellation Plot: {title}")
    plot_filename = f"plot_constellation_{filename_suffix}_{time.strftime('%Y%m%d_%H%M%S')}.png"
    fig = plt.figure(figsize=(8, 8))
    fig.number # Access figure number

    try:
        if signal is None or len(signal) == 0: logger.warning(f"Skipping constellation plot '{title}': No data."); plt.close(fig); return

        # Subsample for plotting if necessary
        if len(signal) > num_points:
             step = max(1, len(signal) // num_points)
             logger.debug(f"Plotting constellation with step={step} ({num_points} target points)")
             plot_data = signal[::step]
        else:
             plot_data = signal

        # Ensure data is finite
        plot_data_finite = plot_data[np.isfinite(plot_data)]
        num_finite = len(plot_data_finite)
        if num_finite == 0: logger.warning(f"Skipping constellation plot '{title}': No finite data points."); plt.close(fig); return
        logger.debug(f"Plotting {num_finite} finite constellation points.")

        plt.scatter(np.real(plot_data_finite), np.imag(plot_data_finite), s=5, alpha=0.3) # Reduced alpha
        plt.xlabel("In-Phase Amplitude") # X Label
        plt.ylabel("Quadrature Amplitude") # Y Label
        plt.title(title) # Title
        plt.grid(True, which='both', linestyle='--')
        plt.axhline(0, color='grey', lw=0.5)
        plt.axvline(0, color='grey', lw=0.5)
        # Set axis limits dynamically
        max_lim = np.max(np.abs(plot_data_finite)) * 1.2 if num_finite > 0 else 0.1
        if max_lim < 1e-9: max_lim = 0.1 # Ensure sensible limit if signal is tiny
        plt.xlim(-max_lim, max_lim); plt.ylim(-max_lim, max_lim)
        plt.gca().set_aspect('equal', adjustable='box') # Ensure equal aspect ratio
        plt.tight_layout()
        save_plot(fig, plot_filename, plot_dir=plot_dir)
    except Exception as e:
        logger.error(f"Error generating constellation plot '{title}': {e}", exc_info=True)
        if 'fig' in locals() and fig is not None and plt.fignum_exists(fig.number): plt.close(fig)