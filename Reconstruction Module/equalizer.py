# equalizer.py
"""Contains functions for adaptive equalization, including blind CMA (GPU accelerated)."""

import numpy as np
from tqdm import tqdm
import logging # Import logging

# --- Get logger for this module ---
logger = logging.getLogger(__name__)
# --- END Get logger ---

# --- Optional CuPy Import ---
cupy_available = False # Default to False
try:
    import cupy as cp
    try:
        cp.cuda.Device(0).use()
        cupy_available = True
        logger.info("CuPy found and GPU available. Enabling GPU acceleration for equalizer.")
    except cp.cuda.runtime.CUDARuntimeError as e:
        logger.warning(f"CuPy found, but GPU unavailable/error: {e}. Equalizer will run on CPU.")
    except Exception as e_init: # Catch other potential init errors
         logger.warning(f"CuPy found, but error during GPU init: {e_init}. Equalizer will run on CPU.")
except ImportError:
    logger.warning("CuPy not found. Equalizer will run on CPU.")
# --- End CuPy Import ---


# QAM16 Slicer (remains on CPU, only used for reference/debug if needed)
def qam16_slicer(symbol):
    # ... (function remains the same) ...
    # Ensure input is numpy for CPU comparison
    symbol_cpu = cp.asnumpy(symbol) if isinstance(symbol, cp.ndarray) else symbol
    levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    real_idx = np.argmin(np.abs(symbol_cpu.real - levels))
    imag_idx = np.argmin(np.abs(symbol_cpu.imag - levels))
    return levels[real_idx] + 1j * levels[imag_idx]


def calculate_cma_r2(constellation_points):
    """Calculates the R2 constant for CMA based on constellation points (CPU)."""
    if constellation_points is None or len(constellation_points) == 0:
        logger.warning("No constellation points provided for CMA R2 calc, using default approximation.")
        return 1.32 # Approximate value for normalized QAM16
    # Ensure calculation on CPU
    points_cpu = cp.asnumpy(constellation_points) if isinstance(constellation_points, cp.ndarray) else np.asarray(constellation_points)
    expected_abs_sq = np.mean(np.abs(points_cpu)**2)
    expected_abs_pow4 = np.mean(np.abs(points_cpu)**4)
    # Godard CMA (p=2) uses R2 = E[ |d_n|^4 ] / E[ |d_n|^2 ]
    if expected_abs_sq < 1e-15:
         logger.warning("Constellation points have near-zero power for R2 calc.")
         return 1.0
    r2 = expected_abs_pow4 / expected_abs_sq
    logger.info(f"Calculated CMA R2 = {r2:.4f}") # Use INFO level for this result
    return r2 # Return CPU scalar

def cma_equalizer(input_signal, num_taps, mu, constellation=None):
    """
    Applies a Constant Modulus Algorithm (CMA) equalizer (GPU accelerated).

    Args:
        input_signal (np.ndarray): The complex input signal (CPU numpy array).
        num_taps (int): Number of FIR filter taps (should be odd).
        mu (float): Step size (learning rate).
        constellation (np.ndarray, optional): Array of complex constellation points (CPU)
                                              used to calculate R2. If None, uses default.

    Returns:
        np.ndarray: The equalized output signal (CPU numpy array).
                    Returns input unmodified if GPU/CuPy not available or error occurs.
    """
    if not cupy_available:
        logger.warning("--- SKIPPING GPU CMA Equalizer: CuPy/GPU not available ---")
        return input_signal

    logger.info(f"--- Applying GPU CMA Equalizer (Taps={num_taps}, mu={mu}) ---")
    if num_taps % 2 == 0:
        logger.error("Number of taps must be odd for CMA equalizer.")
        return input_signal # Return original if taps invalid
    if not isinstance(input_signal, np.ndarray) or input_signal.ndim != 1:
        logger.error("Invalid input signal format for CMA equalizer.")
        return input_signal
    if len(input_signal) < num_taps:
         logger.error(f"Input signal length ({len(input_signal)}) is less than num_taps ({num_taps}).")
         return input_signal # Cannot operate filter

    n_samples = len(input_signal)
    # Use try-except for GPU memory allocation
    try:
        output_signal_gpu = cp.zeros(n_samples, dtype=cp.complex128)
        weights_gpu = cp.zeros(num_taps, dtype=cp.complex128)
        center_tap_idx = num_taps // 2
        weights_gpu[center_tap_idx] = 1.0 + 0j
    except Exception as alloc_e:
        logger.error(f"Failed to allocate GPU memory for equalizer: {alloc_e}", exc_info=True)
        return input_signal # Cannot proceed if memory allocation fails

    # Calculate R2 constant (on CPU) - keep print as logger.info
    R2 = calculate_cma_r2(constellation) # R2 is a CPU scalar

    # --- Transfer input data to GPU ---
    input_signal_gpu = None # Initialize
    input_padded_gpu = None
    try:
        input_signal_gpu = cp.asarray(input_signal)
        # Use cp.pad for consistency if needed, though slicing works
        input_padded_gpu = cp.pad(input_signal_gpu, (num_taps - 1, 0), 'constant')
    except Exception as e:
        logger.error(f"Error moving input signal to GPU: {e}. Aborting equalization.", exc_info=True)
        # Clean up any allocated GPU arrays before returning
        if 'output_signal_gpu' in locals(): del output_signal_gpu
        if 'weights_gpu' in locals(): del weights_gpu
        if 'input_signal_gpu' in locals() and input_signal_gpu is not None: del input_signal_gpu
        return input_signal

    # --- CMA Iteration on GPU ---
    equalized_output_cpu = input_signal # Default to input in case of failure
    try:
        # --- Initialize GPU vars list for cleanup ---
        gpu_vars_in_loop = ['x_buffer_gpu', 'x_rev_gpu', 'y_n_gpu', 'error_term_gpu']

        for n in tqdm(range(n_samples), desc="GPU CMA Equalizing"):
            x_buffer_gpu = input_padded_gpu[n : n + num_taps]
            x_rev_gpu = x_buffer_gpu[::-1]
            y_n_gpu = cp.dot(cp.conj(weights_gpu), x_rev_gpu)
            output_signal_gpu[n] = y_n_gpu
            error_term_gpu = y_n_gpu * (cp.abs(y_n_gpu)**2 - R2)
            weights_gpu = weights_gpu - mu * error_term_gpu * cp.conj(x_rev_gpu)

            # Memory clearing inside loop generally not needed unless extremely large signals/long runs
            # if n % 50000 == 0: cp.get_default_memory_pool().free_all_blocks()

        # --- Transfer final result back to CPU ---
        logger.info("--- CMA Equalization Complete ---")
        equalized_output_cpu = cp.asnumpy(output_signal_gpu)

    except Exception as e:
        logger.error(f"Error during GPU CMA loop at sample {n}: {e}. Returning signal before error.", exc_info=True)
        # Try to return partially processed data if possible
        equalized_output_cpu = cp.asnumpy(output_signal_gpu) # Get whatever was computed
    finally:
        # Explicitly delete large GPU arrays after loop
        logger.debug("Cleaning up GPU memory...")
        del input_signal_gpu, input_padded_gpu, weights_gpu, output_signal_gpu
        # Delete loop vars if they exist from last iteration
        for var_name in gpu_vars_in_loop:
             if var_name in locals() and isinstance(locals()[var_name], cp.ndarray):
                 del locals()[var_name]
        try:
            # Clear memory pool if CuPy available
            if cupy_available:
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
        except Exception as mem_e:
            logger.warning(f"Error freeing CuPy memory pool post-loop: {mem_e}")

    return equalized_output_cpu