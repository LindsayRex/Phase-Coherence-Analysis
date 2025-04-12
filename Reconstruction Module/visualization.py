# visualization.py
"""Functions for plotting results."""

import numpy as np
import matplotlib.pyplot as plt
from . import utils # Use relative import if utils is in the same package

def plot_time_domain(t_vector, gt_signal, recon_signal, target_rms, evm, plot_length):
    """Plots Ground Truth, Reconstructed signal, and Error signal."""
    print("\n--- Generating Matplotlib Time Domain Plot ---")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    plot_samples = min(plot_length, len(t_vector), len(gt_signal), len(recon_signal))

    if plot_samples > 0:
        time_axis_plot = t_vector[:plot_samples] * 1e6 # Time in microseconds

        # Plot GT
        gt_plot_data = gt_signal[:plot_samples]
        axs[0].plot(time_axis_plot, np.real(gt_plot_data), label='GT (Real)', linewidth=1.0)
        axs[0].plot(time_axis_plot, np.imag(gt_plot_data), label='GT (Imag)', alpha=0.7, linewidth=1.0)
        axs[0].set_title(f'Ground Truth (Target RMS={target_rms:.2e})')
        axs[0].set_ylabel('Amplitude'); axs[0].legend(fontsize='small'); axs[0].grid(True, linestyle='--', linewidth=0.5)
        ylim_gt_abs = target_rms * 4
        if np.any(np.isfinite(gt_plot_data)): ylim_gt_abs = max(ylim_gt_abs, np.nanmax(np.abs(gt_plot_data)))
        axs[0].set_ylim(-ylim_gt_abs * 1.2, ylim_gt_abs * 1.2)

        # Plot Recon (Assumed already aligned for plotting)
        recon_plot_data = recon_signal[:plot_samples]
        axs[1].plot(time_axis_plot, np.nan_to_num(np.real(recon_plot_data)), label='Recon (Real)', linewidth=1.0)
        axs[1].plot(time_axis_plot, np.nan_to_num(np.imag(recon_plot_data)), label='Recon (Imag)', alpha=0.7, linewidth=1.0)
        title_recon = f'Reconstructed Signal (Plot Aligned)'
        if np.isfinite(evm): title_recon += f' / Eval EVM: {evm:.2f}%'
        axs[1].set_title(title_recon); axs[1].set_ylabel('Amplitude'); axs[1].legend(fontsize='small'); axs[1].grid(True, linestyle='--', linewidth=0.5)
        ylim_recon_abs = ylim_gt_abs
        if np.any(np.isfinite(recon_plot_data)): ylim_recon_abs = max(ylim_gt_abs, np.nanmax(np.abs(recon_plot_data)))
        axs[1].set_ylim(-ylim_recon_abs * 1.2, ylim_recon_abs * 1.2)

        # Plot Error
        error_signal = gt_plot_data - recon_plot_data # Error relative to the plotted recon signal
        axs[2].plot(time_axis_plot, np.nan_to_num(np.real(error_signal)), label='Error (Real)', linewidth=1.0)
        axs[2].plot(time_axis_plot, np.nan_to_num(np.imag(error_signal)), label='Error (Imag)', alpha=0.7, linewidth=1.0)
        axs[2].set_title('Error Signal (GT - Plot Aligned Recon)'); axs[2].set_xlabel('Time (µs)')
        axs[2].set_ylabel('Amplitude'); axs[2].legend(fontsize='small'); axs[2].grid(True, linestyle='--', linewidth=0.5)
        ylim_err_abs = ylim_recon_abs * 0.5
        if np.any(np.isfinite(error_signal)): ylim_err_abs = max(ylim_err_abs, np.nanmax(np.abs(error_signal)))
        axs[2].set_ylim(-ylim_err_abs * 1.2, ylim_err_abs * 1.2)

        plt.tight_layout(); plt.show(block=False); plt.pause(0.1)
    else: print("Skipping time domain plots: Not enough samples ({plot_samples} samples).")


def plot_spectrum(gt_signal, recon_signal, recon_rate, spectrum_ylim_bottom):
    """Plots the power spectrum comparison."""
    print("\n--- Generating Matplotlib Spectrum Plot ---")
    plt.figure(figsize=(12, 7))
    plot_spectrum_flag = False

    # Plot GT spectrum
    if gt_signal is not None and len(gt_signal) > 1 and recon_rate is not None:
        f_gt, spec_gt_db = utils.compute_spectrum(gt_signal, recon_rate)
        if len(f_gt) > 0: plt.plot(f_gt / 1e6, spec_gt_db, label='GT Spectrum', alpha=0.8, linewidth=1.5); plot_spectrum_flag = True
        else: print("GT spectrum computation returned empty.")
    else: print("Skipping GT spectrum plot (invalid input).")

    # Plot Reconstructed spectrum (using the signal passed, assumed aligned for plotting)
    if recon_signal is not None and len(recon_signal) > 1 and recon_rate is not None:
        f_recon, spec_recon_db = utils.compute_spectrum(recon_signal, recon_rate)
        if len(f_recon) > 0: plt.plot(f_recon / 1e6, spec_recon_db, label='Recon Spectrum (Plot Aligned)', ls='--', alpha=0.9, linewidth=1.5); plot_spectrum_flag = True
        else: print("Reconstructed spectrum computation returned empty.")
    else: print("Skipping Reconstructed spectrum plot (invalid input).")

    if plot_spectrum_flag:
        plt.title('Power Spectrum Comparison')
        plt.xlabel('Frequency (MHz)'); plt.ylabel('Power Spectrum (dB)')
        plt.ylim(bottom=spectrum_ylim_bottom)
        max_spec_val = -np.inf; ax = plt.gca()
        for line in ax.get_lines(): ydata = line.get_ydata(); max_spec_val = max(max_spec_val, np.max(ydata[np.isfinite(ydata)])) if len(ydata)>0 and np.any(np.isfinite(ydata)) else max_spec_val
        if np.isfinite(max_spec_val): plt.ylim(bottom=spectrum_ylim_bottom, top=min(max_spec_val + 10, 20))
        else: plt.ylim(bottom=spectrum_ylim_bottom, top=0)
        plt.legend(); plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.tight_layout(); plt.show() # Show final plot blocking
    else: print("Skipping spectrum plot.")