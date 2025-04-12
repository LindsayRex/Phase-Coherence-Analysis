# data_loader.py
"""Handles loading chunk data and metadata from HDF5 files."""

import h5py
import numpy as np
import sys
import os
import logging # Import standard logging

# --- VVVVV Get logger for this module VVVVV ---
logger = logging.getLogger(__name__)
# --- ^^^^^ Get logger for this module ^^^^^ ---
# DO NOT call log_config.setup_logging() here


def load_data(filename):
    """
    Loads chunk data, metadata, and global attributes from an HDF5 file.

    Args:
        filename (str): Path to the HDF5 input file. If just a filename,
                        looks relative to CWD or SIMULATED_DATA_DIR env var.

    Returns:
        tuple: (loaded_chunks, loaded_metadata, global_attrs)
               Returns (None, None, None) or ([], [], {}) on error/no data.
    """
    filepath = filename # Use this variable for the final path
    # Construct full path if needed
    if not os.path.isabs(filename):
        simulated_data_dir = os.environ.get('SIMULATED_DATA_DIR')
        if simulated_data_dir and os.path.isdir(simulated_data_dir):
            filepath = os.path.join(simulated_data_dir, filename)
            logger.debug(f"Using SIMULATED_DATA_DIR: {simulated_data_dir}")
        else:
            default_rel_path = os.path.join('..', 'Simulated_data')
            filepath_rel = os.path.abspath(os.path.join(os.path.dirname(__file__), default_rel_path, filename))
            filepath_cwd = os.path.abspath(os.path.join(os.getcwd(), filename))
            if os.path.exists(filepath_rel):
                 filepath = filepath_rel
                 logger.debug(f"Using relative path from script location: {filepath}")
            elif os.path.exists(filepath_cwd):
                 filepath = filepath_cwd
                 logger.debug(f"Using path relative to CWD: {filepath}")
            else:
                 logger.warning(f"File '{filename}' not absolute and SIMULATED_DATA_DIR not set/valid. "
                                f"Attempting to open as-is.")
                 # filepath remains the original relative filename

    logger.info(f"Attempting to load data from resolved path: {filepath}")
    loaded_chunks = []
    loaded_metadata = []
    global_attrs = {}
    try:
        with h5py.File(filepath, 'r') as f:
            logger.info("--- Verifying Loaded Metadata Phase ---")
            actual_chunks_meta = f.attrs.get('actual_num_chunks_saved', 0)
            logger.debug(f"File attribute 'actual_num_chunks_saved' = {actual_chunks_meta}")
            true_phases_rad = []
            for i in range(int(actual_chunks_meta)): # Cast to int for range
                group_name = f'chunk_{i:03d}'
                if group_name in f:
                    meta_group = f[group_name]
                    phase_rad = meta_group.attrs.get('applied_phase_offset_rad', np.nan)
                    true_phases_rad.append(phase_rad)
                    phase_deg = np.rad2deg(phase_rad) if np.isfinite(phase_rad) else 'MISSING/NaN'
                    # Log phase info at DEBUG level for less console noise if root is INFO
                    logger.debug(f"Chunk {i}: Metadata 'applied_phase_offset_rad' = {phase_rad} (~{phase_deg} deg)")
                else:
                    logger.warning(f"Chunk {i}: Group '{group_name}' not found during metadata check.")
                    true_phases_rad.append(np.nan)

            logger.info("--- Loading Global Parameters ---")
            for key, value in f.attrs.items():
                global_attrs[key] = value
                logger.debug(f"Global Attr: {key}: {value}")
            logger.info("---------------------------------")

            logger.info("--- Loading Chunk Data and Metadata ---")
            actual_chunks_saved = global_attrs.get('actual_num_chunks_saved', 0)
            if actual_chunks_saved == 0:
                logger.warning("Global attribute 'actual_num_chunks_saved' is 0.")

            num_loaded = 0
            for i in range(int(actual_chunks_saved)): # Cast to int
                group_name = f'chunk_{i:03d}'
                if group_name in f:
                    try:
                        group = f[group_name]
                        # Check if 'iq_data' dataset exists
                        if 'iq_data' not in group:
                             logger.warning(f"Dataset 'iq_data' not found in group '{group_name}'. Skipping chunk.")
                             continue
                        chunk_data = group['iq_data'][:].astype(np.complex128)
                        meta = {key: value for key, value in group.attrs.items()}
                        loaded_chunks.append(chunk_data)
                        loaded_metadata.append(meta)
                        num_loaded += 1
                        logger.debug(f"Loaded chunk {i}: {len(chunk_data)} samples.")
                    except Exception as chunk_e:
                         logger.error(f"Error loading data from group '{group_name}': {chunk_e}", exc_info=False)
                else:
                    logger.warning(f"Chunk group '{group_name}' not found during data loading (Metadata claimed {actual_chunks_saved}).")

            logger.info(f"Finished loading loop. Successfully loaded {num_loaded} chunks.")
            if num_loaded != actual_chunks_saved:
                 logger.warning(f"Mismatch between actual_num_chunks_saved ({actual_chunks_saved}) and chunks loaded ({num_loaded}).")
                 global_attrs['actual_num_chunks_saved'] = num_loaded # Update global attr

    except FileNotFoundError:
        logger.critical(f"HDF5 file not found at path: '{filepath}'")
        return None, None, None
    except Exception as e:
        logger.critical(f"Error loading HDF5 file '{filepath}': {e}", exc_info=True)
        return None, None, None

    if not loaded_chunks:
        logger.error("No chunk data was successfully loaded from file.")
        return [], [], global_attrs # Return empty lists but potentially loaded attrs

    logger.info(f"Successfully loaded {len(loaded_chunks)} chunks from HDF5.")
    return loaded_chunks, loaded_metadata, global_attrs

def validate_global_attrs(global_attrs):
    """Validates essential keys in the global attributes using logging."""
    logger.info("--- Validating Essential Global Attributes ---")
    valid = True
    sdr_rate = global_attrs.get('sdr_sample_rate_hz', None)
    recon_rate = global_attrs.get('ground_truth_sample_rate_hz', None)

    if sdr_rate is None:
        logger.error("Validation Failed: 'sdr_sample_rate_hz' missing.")
        valid = False
    elif not isinstance(sdr_rate, (int, float, np.number)) or sdr_rate <= 0: # Added np.number check for h5py types
         logger.error(f"Validation Failed: Invalid SDR sample rate found: {sdr_rate} (type: {type(sdr_rate)})")
         valid = False
    else:
         logger.debug(f"SDR Sample Rate OK: {sdr_rate}")

    if recon_rate is None:
        logger.error("Validation Failed: 'ground_truth_sample_rate_hz' missing.")
        valid = False
    elif not isinstance(recon_rate, (int, float, np.number)) or recon_rate <= 0:
         logger.error(f"Validation Failed: Invalid reconstruction sample rate found: {recon_rate} (type: {type(recon_rate)})")
         valid = False
    else:
        logger.debug(f"Recon Sample Rate OK: {recon_rate}")

    if 'overlap_factor' not in global_attrs: logger.warning("Validation Info: 'overlap_factor' not found.")
    if 'tuning_delay_s' not in global_attrs: logger.warning("Validation Info: 'tuning_delay_s' not found.")
    if 'pilot_tone_added' not in global_attrs: logger.warning("Validation Info: 'pilot_tone_added' flag missing.")

    if valid: logger.info("Essential global attributes look OK.")
    else: logger.error("Essential global attribute validation FAILED.")
    return valid