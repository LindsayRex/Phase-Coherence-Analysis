# data_loader.py
"""Handles loading chunk data, metadata, global attributes, and ground truth from HDF5 files."""

import h5py
import numpy as np
import sys
import os
import logging

# Get logger for this module
logger = logging.getLogger(__name__)
# DO NOT call log_config.setup_logging() here

def load_data(filename):
    """
    Loads chunk data, metadata, global attributes, and ground truth baseband
    from an HDF5 file.

    Args:
        filename (str): Path to the HDF5 input file. If just a filename,
                        looks relative to CWD or default path.

    Returns:
        tuple: (loaded_chunks, loaded_metadata, global_attrs, gt_baseband, gt_fs_sim)
               - loaded_chunks (list): List of numpy arrays (complex128) of SDR rate chunk data.
               - loaded_metadata (list): List of dictionaries containing metadata for each chunk.
               - global_attrs (dict): Dictionary containing global attributes.
               - gt_baseband (np.ndarray or None): Ground truth baseband signal (at sim rate).
               - gt_fs_sim (float or None): Sample rate of gt_baseband (Hz).
        Returns (None, None, None, None, None) on critical file error.
        Returns ([], [], {}, None, None) if GT is missing but chunks load.
    """
    filepath = filename
    # Construct full path if needed
    if not os.path.isabs(filename):
        # Check standard relative path first (../Simulated_data)
        default_rel_path = os.path.join('..', 'Simulated_data')
        filepath_rel = os.path.abspath(os.path.join(os.path.dirname(__file__), default_rel_path, filename))
        # Check relative to Current Working Directory second
        filepath_cwd = os.path.abspath(os.path.join(os.getcwd(), filename))

        if os.path.exists(filepath_rel):
             filepath = filepath_rel
             logger.debug(f"Using relative path from script location: {filepath}")
        elif os.path.exists(filepath_cwd):
             filepath = filepath_cwd
             logger.debug(f"Using path relative to CWD: {filepath}")
        else:
             logger.warning(f"File '{filename}' relative path not found. Trying to open as-is.")
             # filepath remains the original relative filename

    logger.info(f"Attempting to load data from resolved path: {filepath}")
    loaded_chunks = []
    loaded_metadata = []
    global_attrs = {}
    gt_baseband = None
    gt_fs_sim = None

    try:
        with h5py.File(filepath, 'r') as f:
            # --- Load Global Attrs ---
            logger.info("--- Loading Global Parameters ---")
            for key, value in f.attrs.items(): global_attrs[key] = value; logger.debug(f"Global Attr: {key}: {value}")
            logger.info("---------------------------------")

            # --- VVVVV Load Ground Truth Baseband VVVVV ---
            logger.info("--- Loading Ground Truth Baseband Signal ---")
            if 'ground_truth_baseband' in f:
                gt_baseband_dset = f['ground_truth_baseband']
                gt_baseband = gt_baseband_dset[:].astype(np.complex128)
                # Try getting sample rate from dataset attribute first
                gt_fs_sim = gt_baseband_dset.attrs.get('sample_rate_hz')
                if gt_fs_sim is None:
                    # Fallback to global attribute
                    gt_fs_sim = global_attrs.get('ground_truth_sample_rate_hz_sim')
                    if gt_fs_sim is not None:
                        logger.warning("GT sample rate missing from dataset attrs, using global attr 'ground_truth_sample_rate_hz_sim'.")
                if gt_fs_sim is None:
                     logger.error("Ground truth sample rate ('sample_rate_hz' or 'ground_truth_sample_rate_hz_sim') NOT FOUND!")
                     # Continue loading chunks, but return None for GT rate
                else:
                     logger.info(f"Loaded ground truth baseband: {len(gt_baseband)} samples @ {gt_fs_sim/1e6:.2f} MHz")
            else:
                logger.error("Dataset 'ground_truth_baseband' not found in HDF5 file. Cannot perform evaluation.")
                # Continue loading chunks, but GT will be None
            # --- ^^^^^ Load Ground Truth Baseband ^^^^^ ---


            # --- Load Chunks and Metadata ---
            logger.info("--- Loading Chunk Data and Metadata ---")
            actual_chunks_saved = global_attrs.get('actual_num_chunks_saved', 0)
            if actual_chunks_saved == 0: logger.warning("Global attribute 'actual_num_chunks_saved' is 0.")

            num_loaded = 0
            for i in range(int(actual_chunks_saved)):
                group_name = f'chunk_{i:03d}';
                if group_name in f:
                    try:
                        group = f[group_name]
                        if 'iq_data' not in group: logger.warning(f"No 'iq_data' in {group_name}. Skipping."); continue
                        chunk_data = group['iq_data'][:].astype(np.complex128)
                        meta = {key: value for key, value in group.attrs.items()}
                        loaded_chunks.append(chunk_data); loaded_metadata.append(meta); num_loaded += 1
                        logger.debug(f"Loaded chunk {i}: {len(chunk_data)} samples.")
                    except Exception as chunk_e: logger.error(f"Error loading group '{group_name}': {chunk_e}", exc_info=False)
                else: logger.warning(f"Chunk group '{group_name}' not found (Metadata claimed {actual_chunks_saved}).")

            logger.info(f"Finished loading loop. Successfully loaded {num_loaded} chunks.")
            if num_loaded != actual_chunks_saved:
                 logger.warning(f"Mismatch: actual_num_chunks_saved={actual_chunks_saved}, loaded={num_loaded}.")
                 global_attrs['actual_num_chunks_saved'] = num_loaded

    except FileNotFoundError:
        logger.critical(f"HDF5 file not found at path: '{filepath}'")
        return None, None, None, None, None # Return all Nones on file not found
    except Exception as e:
        logger.critical(f"Error loading HDF5 file '{filepath}': {e}", exc_info=True)
        return None, None, None, None, None # Return all Nones on other critical errors

    if not loaded_chunks: logger.warning("No chunk data was successfully loaded from file.")
    if gt_baseband is None: logger.warning("Ground truth baseband was not loaded successfully.")

    logger.info(f"Data loading complete. Loaded {len(loaded_chunks)} chunks.")
    # Return GT signal and its sample rate
    return loaded_chunks, loaded_metadata, global_attrs, gt_baseband, gt_fs_sim


def validate_global_attrs(global_attrs):
    """Validates essential keys in the global attributes using logging."""
    logger.info("--- Validating Essential Global Attributes ---")
    valid = True
    sdr_rate = global_attrs.get('sdr_sample_rate_hz', None)
    # Recon rate from config is now the primary target, not necessarily in HDF5 global attrs
    # recon_rate = global_attrs.get('ground_truth_sample_rate_hz', None) # Comment out check for this old key

    if sdr_rate is None: logger.error("Validation Failed: 'sdr_sample_rate_hz' missing."); valid = False
    elif not isinstance(sdr_rate, (int, float, np.number)) or sdr_rate <= 0: logger.error(f"Validation Failed: Invalid SDR sample rate: {sdr_rate}"); valid = False
    else: logger.debug(f"SDR Sample Rate OK: {sdr_rate}")

    # Check for GT sim rate used by generator
    gt_sim_rate = global_attrs.get('ground_truth_sample_rate_hz_sim', None)
    if gt_sim_rate is None: logger.warning("Validation Info: 'ground_truth_sample_rate_hz_sim' missing (needed for GT resampling).")
    elif not isinstance(gt_sim_rate, (int, float, np.number)) or gt_sim_rate <= 0: logger.warning(f"Validation Info: Invalid GT Sim Rate: {gt_sim_rate}")
    else: logger.debug(f"GT Sim Rate OK: {gt_sim_rate}")


    # Optional checks with warnings
    if 'overlap_factor' not in global_attrs: logger.warning("Validation Info: 'overlap_factor' not found.")
    if 'tuning_delay_s' not in global_attrs: logger.warning("Validation Info: 'tuning_delay_s' not found.")
    if 'pilot_tone_added' not in global_attrs: logger.warning("Validation Info: 'pilot_tone_added' flag missing.")

    if valid: logger.info("Essential global attributes look OK.")
    else: logger.error("Essential global attribute validation FAILED.")
    return valid