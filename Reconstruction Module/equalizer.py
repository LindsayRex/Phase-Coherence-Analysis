# equalizer.py
"""Contains functions for adaptive equalization, including blind CMA (GPU accelerated)."""

import numpy as np
from tqdm import tqdm

# --- Optional CuPy Import ---
try:
    import cupy as cp
    # Check if a GPU is available and select the default one
    try:
        cp.cuda.Device(0).use()
        cupy_available = True
        print("CuPy found and GPU available. Enabling GPU acceleration for equalizer.")
    except cp.cuda.runtime.CUDARuntimeError as e:
        print(f"Warning: CuPy found, but GPU unavailable/error: {e}. Equalizer will run on CPU.")
        cupy_available = False
except ImportError:
    print("Warning: CuPy not found. Equalizer will run on CPU.")
    cupy_available = False
# --- End CuPy Import ---


# QAM16 Slicer (remains on CPU, only used for reference/debug if needed)
def qam16_slicer(symbol):
    # Ensure input is numpy for CPU comparison
    symbol_cpu = cp.asnumpy(symbol) if isinstance(symbol, cp.ndarray) else symbol
    levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    real_idx = np.argmin(np.abs(symbol_cpu.real - levels))
    imag_idx = np.argmin(np.abs(symbol_cpu.imag - levels))
    return levels[real_idx] + 1j * levels[imag_idx]

def calculate_cma_r2(constellation_points):
    """Calculates the R2 constant for CMA based on constellation points (CPU)."""
    if constellation_points is None or len(constellation_points) == 0:
        print("Warning: No constellation points provided for CMA R2 calc, using default approximation.")
        return 1.32 # Approximate value for normalized QAM16
    # Ensure calculation on CPU
    points_cpu = cp.asnumpy(constellation_points) if isinstance(constellation_points, cp.ndarray) else np.asarray(constellation_points)
    expected_abs_sq = np.mean(np.abs(points_cpu)**2)
    expected_abs_pow4 = np.mean(np.abs(points_cpu)**4)
    # Godard CMA (p=2) uses R2 = E[ |d_n|^4 ] / E[ |d_n|^2 ]
    if expected_abs_sq < 1e-15: # Avoid division by near-zero
         print("Warning: Constellation points have near-zero power for R2 calc.")
         return 1.0
    r2 = expected_abs_pow4 / expected_abs_sq
    print(f"Calculated CMA R2 = {r2:.4f}")
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
                    Returns input unmodified if GPU/CuPy not available.
    """
    if not cupy_available:
        print("--- SKIPPING GPU CMA Equalizer: CuPy/GPU not available ---")
        return input_signal # Return input unmodified

    print(f"\n--- Applying GPU CMA Equalizer (Taps={num_taps}, mu={mu}) ---")
    if num_taps % 2 == 0:
        print("Error: Number of taps must be odd.")
        return input_signal
    if not isinstance(input_signal, np.ndarray) or input_signal.ndim != 1:
        print("Error: Invalid input signal format.")
        return input_signal

    n_samples = len(input_signal)
    output_signal_gpu = cp.zeros(n_samples, dtype=cp.complex128) # Output on GPU
    weights_gpu = cp.zeros(num_taps, dtype=cp.complex128)      # Weights on GPU
    center_tap_idx = num_taps // 2
    weights_gpu[center_tap_idx] = 1.0 + 0j # Initialize center tap on GPU

    # Calculate R2 constant (on CPU)
    R2 = calculate_cma_r2(constellation) # R2 is a CPU scalar

    # --- Transfer input data to GPU ---
    try:
        input_signal_gpu = cp.asarray(input_signal)
        input_padded_gpu = cp.pad(input_signal_gpu, (num_taps - 1, 0), 'constant')
    except Exception as e:
        print(f"Error moving input signal to GPU: {e}. Aborting equalization.")
        return input_signal # Return original on transfer error

    # --- CMA Iteration on GPU ---
    try:
        for n in tqdm(range(n_samples), desc="GPU CMA Equalizing"):
            # Get buffer slice from padded input on GPU
            x_buffer_gpu = input_padded_gpu[n : n + num_taps]
            x_rev_gpu = x_buffer_gpu[::-1] # Reverse on GPU

            # Filter output: y[n] = w^H * x_rev
            y_n_gpu = cp.dot(cp.conj(weights_gpu), x_rev_gpu)
            output_signal_gpu[n] = y_n_gpu # Store output on GPU

            # CMA Error term (using R2 CPU scalar)
            error_term_gpu = y_n_gpu * (cp.abs(y_n_gpu)**2 - R2)

            # Weight Update: w[n+1] = w[n] - mu * error_term * conj(x_rev)
            weights_gpu = weights_gpu - mu * error_term_gpu * cp.conj(x_rev_gpu)

            # Potential intermediate memory clearing if needed for very long signals
            # if n % 10000 == 0:
            #     cp.get_default_memory_pool().free_all_blocks()

    except Exception as e:
        print(f"Error during GPU CMA loop at sample {n}: {e}. Returning partially processed signal.")
        # Transfer whatever was computed back to CPU
        return cp.asnumpy(output_signal_gpu)
    finally:
        # Explicitly delete large GPU arrays after loop if needed
        del input_signal_gpu, input_padded_gpu, weights_gpu
        del x_buffer_gpu, x_rev_gpu, y_n_gpu, error_term_gpu
        try:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        except Exception as mem_e:
            print(f"Warning: Error freeing CuPy memory pool post-loop: {mem_e}")

    # --- Transfer final result back to CPU ---
    print("--- CMA Equalization Complete ---")
    output_signal_cpu = cp.asnumpy(output_signal_gpu)
    del output_signal_gpu # Free GPU memory
    try:
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
    except Exception as mem_e:
        print(f"Warning: Error freeing CuPy memory pool after final transfer: {mem_e}")

    return output_signal_cpu