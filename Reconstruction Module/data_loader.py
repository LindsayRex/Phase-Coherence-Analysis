# data_loader.py
"""Handles loading chunk data and metadata from HDF5 files."""

import h5py
import numpy as np
import sys
import os  # Import the os module

import logging
# --- Setup Logging (Call setup function from log_config) ---
from . import log_config
# Configure logging level here (e.g., logging.INFO or logging.DEBUG)
log_config.setup_logging(level=logging.DEBUG, log_dir="run_logs")
# --- End Logging Setup ---


def load_data(filename):
    """
    Loads chunk data, metadata, and global attributes from an HDF5 file.

    Args:
        filename (str): Path to the HDF5 input file.  If just a filename is provided,
                        it will look in the SIMULATED_DATA_DIR environment variable.

    Returns:
        tuple: (loaded_chunks, loaded_metadata, global_attrs)
               - loaded_chunks (list): List of numpy arrays (complex128) containing IQ data.
               - loaded_metadata (list): List of dictionaries containing metadata for each chunk.
               - global_attrs (dict): Dictionary containing global attributes from the HDF5 file.
        Returns (None, None, None) on error.
    """
    # Check if the filename is an absolute path or just a filename
    if not os.path.isabs(filename):
        # If it's just a filename, prepend the SIMULATED_DATA_DIR
        simulated_data_dir = os.environ.get('SIMULATED_DATA_DIR', 'simulated_data')
        filename = os.path.join(simulated_data_dir, filename)

    print(f"Loading data from: {filename}")
    loaded_chunks = []
    loaded_metadata = []
    global_attrs = {}
    try:
        with h5py.File(filename, 'r') as f:
            # --- Verify Metadata Phase (Optional but good practice) ---
            print("\n--- Verifying Loaded Metadata Phase ---")
            actual_chunks_meta = f.attrs.get('actual_num_chunks_saved', 0)
            true_phases_rad = [] # Store for potential future use/debugging
            for i in range(actual_chunks_meta):
                group_name = f'chunk_{i:03d}'
                if group_name in f:
                    meta_group = f[group_name]
                    phase_rad = meta_group.attrs.get('applied_phase_offset_rad', np.nan)
                    true_phases_rad.append(phase_rad)
                    phase_deg = np.rad2deg(phase_rad) if np.isfinite(phase_rad) else 'MISSING or NaN'
                    print(f"Chunk {i}: loaded applied_phase_offset_rad = {phase_rad} (approx {phase_deg} deg)")
                else:
                    print(f"Chunk {i}: Group not found during metadata check.")
                    true_phases_rad.append(np.nan)

            # --- Load Global Attrs ---
            for key, value in f.attrs.items():
                global_attrs[key] = value
            print("\n--- Global Parameters ---")
            for key, value in global_attrs.items():
                print(f"{key}: {value}")
            print("-------------------------")

            # --- Load Chunks and Full Metadata ---
            actual_chunks_saved = global_attrs.get('actual_num_chunks_saved', 0)
            if actual_chunks_saved == 0:
                print("Warning: No chunks reported saved in global attributes.")
                # Continue, might load if groups exist anyway, or fail later

            for i in range(actual_chunks_saved): # Iterate based on metadata attribute
                group_name = f'chunk_{i:03d}'
                if group_name in f:
                    group = f[group_name]
                    chunk_data = group['iq_data'][:].astype(np.complex128)
                    meta = {key: value for key, value in group.attrs.items()}
                    loaded_chunks.append(chunk_data)
                    loaded_metadata.append(meta)
                else:
                    print(f"Warning: Chunk group '{group_name}' not found during data loading despite metadata claim. Skipping.")
                    # Append placeholders if strict alignment is needed downstream
                    # loaded_chunks.append(np.array([], dtype=np.complex128))
                    # loaded_metadata.append({})

    except FileNotFoundError:
        print(f"Error: Input HDF5 file not found at '{filename}'")
        return None, None, None
    except Exception as e:
        print(f"Error loading HDF5 file '{filename}': {e}")
        return None, None, None

    if not loaded_chunks:
        print("Error: No chunk data was successfully loaded.")
        # Return empty lists/dict instead of None if preferred downstream
        return [], [], global_attrs # Allow partial success with metadata?
        # Or return None, None, None for hard failure

    print(f"\nSuccessfully loaded {len(loaded_chunks)} chunks.")
    return loaded_chunks, loaded_metadata, global_attrs

def validate_global_attrs(global_attrs):
    """Validates essential keys in the global attributes."""
    sdr_rate = global_attrs.get('sdr_sample_rate_hz', None)
    recon_rate = global_attrs.get('ground_truth_sample_rate_hz', None)

    if sdr_rate is None or recon_rate is None:
        print("Error: Sample rate information ('sdr_sample_rate_hz' or 'ground_truth_sample_rate_hz') missing in global attributes.")
        return False
    if not isinstance(sdr_rate, (int, float)) or sdr_rate <= 0:
         print(f"Error: Invalid SDR sample rate found: {sdr_rate}")
         return False
    if not isinstance(recon_rate, (int, float)) or recon_rate <= 0:
         print(f"Error: Invalid reconstruction sample rate found: {recon_rate}")
         return False

    # Add checks for other essential keys like overlap_factor, tuning_delay_s if needed
    if 'overlap_factor' not in global_attrs: print("Warning: 'overlap_factor' not found in global attributes.")
    if 'tuning_delay_s' not in global_attrs: print("Warning: 'tuning_delay_s' not found in global attributes.")

    return True