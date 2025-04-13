# equalizer.py
"""Contains functions for adaptive equalization, including optimized batch CMA (GPU accelerated)."""

import numpy as np
from tqdm import tqdm
import logging

# --- Get logger for this module ---
logger = logging.getLogger(__name__)
# --- End Get logger ---

# --- Optional CuPy Import ---
cupy_available = False
try:
    import cupy as cp
    try:
        cp.cuda.Device(0).use()
        cupy_available = True
        logger.info("CuPy found and GPU available. Enabling GPU acceleration for equalizer.")
    except cp.cuda.runtime.CUDARuntimeError as e: 
        logger.warning(f"CuPy GPU unavailable: {e}. Equalizer on CPU.")
    except Exception as e_init: 
        logger.warning(f"CuPy GPU init error: {e_init}. Equalizer on CPU.")
except ImportError: 
    logger.warning("CuPy not found. Equalizer on CPU.")
# --- End CuPy Import ---


# QAM16 Slicer (remains on CPU, only used for reference/debug if needed)
def qam16_slicer(symbol):
    symbol_cpu = cp.asnumpy(symbol) if isinstance(symbol, cp.ndarray) else symbol
    levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    real_idx = np.argmin(np.abs(symbol_cpu.real - levels))
    imag_idx = np.argmin(np.abs(symbol_cpu.imag - levels))
    return levels[real_idx] + 1j * levels[imag_idx]


def calculate_cma_r2(constellation_points):
    """Calculates the R2 constant for CMA based on constellation points (CPU)."""
    if constellation_points is None or len(constellation_points) == 0:
        logger.warning("No constellation points provided for CMA R2 calc, using default QAM16 value.")
        # R2 = E[|d|^4] / E[|d|^2]
        # For QAM16 normalized to E[|d|^2]=1, R2 = 1.32
        return 1.32
    points_cpu = cp.asnumpy(constellation_points) if isinstance(constellation_points, cp.ndarray) else np.asarray(constellation_points)
    expected_abs_sq = np.mean(np.abs(points_cpu)**2)
    expected_abs_pow4 = np.mean(np.abs(points_cpu)**4)
    if expected_abs_sq < 1e-15: 
        logger.warning("Constellation points have near-zero power.")
        return 1.0
    r2 = expected_abs_pow4 / expected_abs_sq  # Godard definition
    logger.info(f"Calculated CMA R2 = {r2:.4f} (using provided constellation)")
    return r2


def cma_equalizer_optimized(input_signal, num_taps, mu, constellation=None, log_interval=50000, 
                          batch_size=1024, precision='float64', memory_efficient=True):
    """
    Applies a Constant Modulus Algorithm (CMA) equalizer with batch processing (GPU accelerated).
    
    Args:
        input_signal (np.ndarray): Complex input signal (CPU numpy array).
        num_taps (int): Number of FIR filter taps (should be odd).
        mu (float): Step size (learning rate).
        constellation (np.ndarray, optional): Constellation points (CPU) for R2 calc.
        log_interval (int): How often (in samples) to log convergence stats.
        batch_size (int): Number of samples to process in parallel.
        precision (str): 'float64' for complex128 or 'float32' for complex64.
        memory_efficient (bool): Whether to use a more memory-efficient algorithm.
        
    Returns:
        np.ndarray: Equalized output signal (CPU numpy array). Returns input if error.
    """
    if not cupy_available: 
        logger.warning("--- SKIPPING GPU CMA: CuPy/GPU unavailable ---")
        return input_signal

    # Validate inputs
    if num_taps % 2 == 0: 
        logger.error("Num taps must be odd.")
        return input_signal
    if not isinstance(input_signal, np.ndarray) or input_signal.ndim != 1: 
        logger.error("Invalid input signal format.")
        return input_signal
    if len(input_signal) < num_taps: 
        logger.error(f"Input len ({len(input_signal)}) < num_taps ({num_taps}).")
        return input_signal
    
    # Set precision
    dtype = cp.complex128 if precision == 'float64' else cp.complex64
    logger.info(f"--- Applying GPU Batch CMA Equalizer (Taps={num_taps}, mu={mu:.2e}, "
               f"batch={batch_size}, {precision}) ---")
    
    n_samples = len(input_signal)
    batch_size = min(batch_size, n_samples)  # Ensure batch size isn't larger than input
    
    # Calculate R2 (CPU scalar)
    R2 = calculate_cma_r2(constellation)
    logger.info(f"Using R2={R2:.4f} for CMA cost function.")
    
    # Initialize on GPU
    try:
        # Initialize weights with center spike
        weights_gpu = cp.zeros(num_taps, dtype=dtype)
        center_tap_idx = num_taps // 2
        weights_gpu[center_tap_idx] = 1.0 + 0j  # Center spike initialization
        
        # Transfer input to GPU
        input_signal_gpu = cp.asarray(input_signal, dtype=dtype)
        input_padded_gpu = cp.pad(input_signal_gpu, (num_taps - 1, 0), 'constant')
        
        # Allocate output
        output_signal_gpu = cp.zeros(n_samples, dtype=dtype)
        
        logger.debug(f"Input signal transferred to GPU ({input_signal_gpu.nbytes / 1e6:.1f} MB).")
    except Exception as e:
        logger.error(f"Error setting up GPU arrays: {e}", exc_info=True)
        return input_signal
    
    # Memory efficient approach with vectorized operations
    if memory_efficient:
        try:
            # Process in batches for better GPU utilization
            n_batches = int(np.ceil(n_samples / batch_size))
            last_logged_batch = -1  # Force log on first batch
            
            for batch_idx in tqdm(range(n_batches), desc="GPU CMA Equalizing (Batched)", mininterval=1.0):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, n_samples)
                current_batch_size = batch_end - batch_start
                
                # Vectorized batch_segments creation using advanced indexing
                # Create indices for the sliding window
                indices = cp.arange(batch_start, batch_end)[:, None] + cp.arange(num_taps)[::-1][None, :]
                batch_segments = input_padded_gpu[indices]  # Shape: (current_batch_size, num_taps)
                
                # Batch compute outputs: y = X * w (matrix multiplication for all samples at once)
                # Shape: (batch_size, num_taps) @ (num_taps,) -> (batch_size,)
                y_batch = cp.dot(batch_segments, cp.conj(weights_gpu))
                
                # Store outputs
                output_signal_gpu[batch_start:batch_end] = y_batch
                
                # Batch compute errors and update weights (vectorized)
                abs_y_squared = cp.abs(y_batch)**2
                cost_factors = abs_y_squared - R2  # Shape: (batch_size,)
                
                # Vectorized weight update using einsum to compute gradients
                # error_term = y_batch * cost_factors: Shape (batch_size,)
                # Gradient for each sample: error_term[i] * conj(batch_segments[i])
                # We want: weights -= mu * sum(error_term[i] * conj(batch_segments[i]))
                error_terms = y_batch * cost_factors  # Shape: (batch_size,)
                # einsum: 'i,ij->j' computes sum over batch dimension
                gradient = cp.einsum('i,ij->j', error_terms, cp.conj(batch_segments))
                weights_gpu -= mu * gradient
                
                # Log stats periodically
                current_sample = batch_end
                if (current_sample - batch_start > 0 and 
                    (batch_idx - last_logged_batch) * batch_size >= log_interval or batch_idx == n_batches - 1):
                    weights_norm = cp.linalg.norm(weights_gpu)
                    avg_error = cp.mean(cp.abs(cost_factors))
                    logger.debug(f"  CMA Batch {batch_idx+1}/{n_batches} (Sample {current_sample}/{n_samples}): "
                                f"Weights Norm = {weights_norm:.4f}, Avg Error = {avg_error:.6f}")
                    last_logged_batch = batch_idx
                    
                    # Check for divergence
                    if not cp.all(cp.isfinite(weights_gpu)):
                        logger.error(f"CMA diverged at batch {batch_idx+1}! Weights contain NaN/Inf. Check 'mu'.")
                        raise RuntimeError("CMA Divergence")

            logger.info("--- Batch CMA Equalization Complete ---")
            
        except RuntimeError as r_err:
            logger.error(f"Runtime error during CMA batch loop: {r_err}")
        except Exception as e:
            logger.error(f"Error during GPU CMA batch loop: {e}", exc_info=True)
            
    else:
        # Original algorithm but with better memory management - process one sample at a time
        # but with improved buffer handling and less overhead
        try:
            # Pre-allocate buffers to avoid repeated allocations
            x_buffer_gpu = cp.zeros(num_taps, dtype=dtype)
            last_logged_n = -log_interval
            
            for n in tqdm(range(n_samples), desc="GPU CMA Equalizing", mininterval=1.0):
                # Extract buffer without creating new array
                x_buffer_gpu = input_padded_gpu[n:n+num_taps]
                
                # Calculate filter output y[n] = w^H * x (FIR)
                # Using direct dot product with conjugate weights (more efficient)
                y_n_gpu = cp.dot(cp.conj(weights_gpu), x_buffer_gpu[::-1])
                output_signal_gpu[n] = y_n_gpu
                
                # Calculate CMA error term
                cost_gradient_term = y_n_gpu * (cp.abs(y_n_gpu)**2 - R2)
                
                # Update weights (in-place)
                weights_gpu -= mu * cost_gradient_term * cp.conj(x_buffer_gpu[::-1])
                
                # Log convergence periodically
                if n - last_logged_n >= log_interval:
                    weights_norm = cp.linalg.norm(weights_gpu)
                    logger.debug(f"  CMA Iter {n}/{n_samples}: Weights Norm = {weights_norm:.4f}")
                    last_logged_n = n
                    
                    # Check for divergence
                    if not cp.all(cp.isfinite(weights_gpu)):
                        logger.error(f"CMA diverged at iteration {n}! Weights contain NaN/Inf. Check 'mu'.")
                        raise RuntimeError("CMA Divergence")
            
            logger.info("--- CMA Equalization Complete ---")
            
        except RuntimeError as r_err:
            logger.error(f"Runtime error during CMA loop: {r_err}")
        except Exception as e:
            logger.error(f"Error during GPU CMA loop at sample {n}: {e}", exc_info=True)
    
    # Transfer result back to CPU
    try:
        equalized_output_cpu = cp.asnumpy(output_signal_gpu)
    except Exception as e:
        logger.error(f"Error transferring result to CPU: {e}", exc_info=True)
        equalized_output_cpu = input_signal  # Return original on error
    
    # Clean up GPU memory
    try:
        logger.debug("Cleaning up CMA GPU memory...")
        del input_signal_gpu, input_padded_gpu, weights_gpu, output_signal_gpu
        
        if 'batch_segments' in locals():
            del batch_segments
        if 'y_batch' in locals():
            del y_batch
        if 'x_buffer_gpu' in locals():
            del x_buffer_gpu
            
        if cupy_available:
            cp.get_default_memory_pool().free_all_blocks()
    except Exception as mem_e:
        logger.warning(f"Error freeing CuPy memory: {mem_e}")
    
    return equalized_output_cpu


# Legacy function name for backward compatibility
def cma_equalizer(input_signal, num_taps, mu, constellation=None, log_interval=50000):
    """
    Legacy wrapper around optimized implementation for backward compatibility.
    """
    return cma_equalizer_optimized(
        input_signal, 
        num_taps, 
        mu, 
        constellation=constellation, 
        log_interval=log_interval,
        batch_size=1024,  # Default batch size
        precision='float64'  # Keep original precision for compatibility
    )