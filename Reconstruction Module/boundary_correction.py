# boundary_correction.py
"""Applies adaptive cancellation at chunk boundaries."""

import numpy as np
from tqdm import tqdm

# --- Optional CuPy Import ---
try:
    import cupy as cp
    # Check if a GPU is available and select the default one
    try:
        cp.cuda.Device(0).use()
        cupy_available = True
        print("CuPy found and GPU available. Enabling GPU for boundary cancellation.")
    except cp.cuda.runtime.CUDARuntimeError as e:
        print(f"Warning: CuPy found, but GPU unavailable/error: {e}. Boundary cancellation will run on CPU.")
        cupy_available = False
except ImportError:
    print("Warning: CuPy not found. Boundary cancellation will run on CPU.")
    cupy_available = False
# --- End CuPy Import ---


def adaptive_boundary_cancellation(chunks, recon_rate, overlap_factor, mu=0.001, num_taps=11, num_iterations_per_sample=5):
    """
    Apply adaptive FIR filter locally at boundaries to minimize discontinuity.

    Args:
        chunks (list): List of upsampled chunks (NumPy arrays).
        recon_rate (float): Reconstruction sample rate (Hz).
        overlap_factor (float): Overlap fraction used for stitching.
        mu (float): LMS step size (start small, e.g., 1e-3 to 1e-5).
        num_taps (int): FIR filter taps (odd number recommended).
        num_iterations_per_sample (int): Adaptation iterations per boundary sample.

    Returns:
        list: corrected_chunks (list of NumPy arrays).
    """
    print(f"\n--- Performing Adaptive Boundary Cancellation (Taps={num_taps}, mu={mu}) ---")

    if not chunks:
        print("Error: No chunks provided.")
        return []

    xp = cp if cupy_available else np
    print(f"Using {'GPU (CuPy)' if cupy_available else 'CPU (NumPy)'} for boundary cancellation.")

    # Calculate overlap samples
    chunk_duration_s = len(chunks[0]) / recon_rate if recon_rate > 0 and len(chunks) > 0 else 0
    if chunk_duration_s <= 0:
         print("Warning: Cannot determine chunk duration for overlap calc.")
         overlap_samples = 0
    else:
         overlap_samples = int(round(chunk_duration_s * overlap_factor * recon_rate))
    print(f"Overlap samples for boundary processing: {overlap_samples}")

    if overlap_samples < num_taps:
         print(f"Warning: Overlap samples ({overlap_samples}) < num_taps ({num_taps}). Skipping boundary cancellation.")
         return chunks # Not enough data to operate filter

    # # Start with the first chunk unchanged (on CPU)
    corrected_chunks_cpu = [np.asarray(chunks[0])] # <--- Initial list defined here

    for i in tqdm(range(1, len(chunks)), desc="Adaptive Cancellation"):
        # Get chunks involved in the boundary (ensure NumPy CPU arrays)
        # --- VVVVV CORRECTED LINE VVVVV ---
        prev_chunk_cpu = np.asarray(corrected_chunks_cpu[-1]) # Use the correct list name
        # --- ^^^^^ CORRECTED LINE ^^^^^ ---
        curr_chunk_original_cpu = np.asarray(chunks[i]) # Use the original current chunk

        # Check lengths
        if len(prev_chunk_cpu) < overlap_samples or len(curr_chunk_original_cpu) < overlap_samples:
            print(f"Warning: Insufficient length for overlap in chunk {i}. Copying chunk.")
            corrected_chunks_cpu.append(curr_chunk_original_cpu) # Append to correct list
            continue

        # Extract overlap regions (CPU)
        prev_overlap_cpu = prev_chunk_cpu[-overlap_samples:]
        curr_overlap_cpu = curr_chunk_original_cpu[:overlap_samples]

        # Initialize filter weights (GPU or CPU)
        w = xp.zeros(num_taps, dtype=xp.complex128)
        # Optional: center tap init? w[num_taps // 2] = 1.0? Start with zero.

        # Prepare GPU data if applicable
        if cupy_available:
            try:
                prev_overlap_gpu = cp.asarray(prev_overlap_cpu)
                curr_overlap_gpu = cp.asarray(curr_overlap_cpu)
                # Create buffer for filter input on GPU
                buffer_gpu = cp.zeros(num_taps, dtype=cp.complex128)
                # Create array to store corrected overlap on GPU
                corrected_overlap_gpu = cp.copy(curr_overlap_gpu) # Start with original
            except Exception as e:
                print(f"Error transferring to GPU for chunk {i}: {e}. Skipping boundary cancellation.")
                corrected_chunks_cpu.append(curr_chunk_original_cpu)
                continue
        else:
            prev_overlap_gpu = prev_overlap_cpu # Alias for clarity
            curr_overlap_gpu = curr_overlap_cpu
            buffer_gpu = np.zeros(num_taps, dtype=np.complex128)
            corrected_overlap_gpu = np.copy(curr_overlap_gpu)


        # --- LMS Adaptation within the overlap region ---
        # Iterate through each sample index 'k' in the overlap of the *current* chunk
        # where we want to apply a correction based on the *previous* chunk's end.
        for k in range(overlap_samples):
            # Define input vector for the filter at this point 'k'
            # Takes samples from prev_chunk ending at index corresponding to k
            # Careful with indexing relative to the full prev_chunk vs overlap array
            start_idx_in_prev = len(prev_chunk_cpu) - overlap_samples + k - (num_taps - 1)
            end_idx_in_prev = len(prev_chunk_cpu) - overlap_samples + k + 1
            if start_idx_in_prev < 0: # Handle beginning boundary condition
                 pad_width = abs(start_idx_in_prev)
                 buffer_cpu = np.pad(prev_chunk_cpu[:end_idx_in_prev],(pad_width,0),'constant')
            else:
                 buffer_cpu = prev_chunk_cpu[start_idx_in_prev:end_idx_in_prev]

            # Ensure buffer has correct size (safety check)
            if len(buffer_cpu) != num_taps: continue # Should not happen if logic is correct

            # Transfer buffer to GPU if needed
            if cupy_available:
                 buffer_gpu = cp.asarray(buffer_cpu)
            else:
                 buffer_gpu = buffer_cpu # Already numpy

            # Adapt multiple times for each sample (optional, for faster convergence)
            # This is non-standard LMS but can sometimes help if data changes slowly
            for _ in range(num_iterations_per_sample):
                # Calculate filter output (correction signal)
                y_correction = xp.dot(w, buffer_gpu) # Simple dot product for FIR

                # Calculate current output including correction attempt
                z_k = curr_overlap_gpu[k] + y_correction

                # Error signal (Option A: Minimize difference to *previous* chunk's sample at equivalent point)
                # Target is the sample from the end of the previous chunk corresponding to k
                target_sample = prev_overlap_gpu[k]
                e = target_sample - z_k # Error = target - current_output

                # LMS Update rule: w = w + mu * error * conj(input_buffer)
                # Use conjugate of input buffer for complex LMS
                w = w + mu * e * xp.conj(buffer_gpu)

            # Apply the *final* correction for this sample 'k' after iterations
            final_y_correction = xp.dot(w, buffer_gpu)
            corrected_overlap_gpu[k] = curr_overlap_gpu[k] + final_y_correction


        # --- Construct the full corrected chunk ---
        # Copy the corrected overlap (back to CPU) into the original current chunk
        corrected_chunk_cpu = curr_chunk_original_cpu.copy()
        corrected_chunk_cpu[:overlap_samples] = cp.asnumpy(corrected_overlap_gpu) if cupy_available else corrected_overlap_gpu

        corrected_chunks_cpu.append(corrected_chunk_cpu) # Add the processed chunk

        # Optional: Store final weights for analysis
        # filter_weights_list.append(cp.asnumpy(w) if cupy_available else w)

        # --- Clear GPU Memory for this iteration ---
        if cupy_available:
            gpu_vars_iter = ['prev_overlap_gpu', 'curr_overlap_gpu', 'w',
                             'buffer_gpu', 'corrected_overlap_gpu',
                             'y_correction', 'z_k', 'target_sample', 'e']
            for var_name in gpu_vars_iter:
                 if var_name in locals() and isinstance(locals()[var_name], cp.ndarray):
                     del locals()[var_name]
            try:
                mempool = cp.get_default_memory_pool().free_all_blocks()
            except Exception as mem_e:
                print(f"Warning: Error freeing CuPy memory pool in loop: {mem_e}")

    print("--- Adaptive Boundary Cancellation Complete ---")
    return corrected_chunks_cpu # Return list of NumPy arrays