# log_config.py
import logging
import os
import time

# --- VVVVV SET DEFAULT LEVEL TO DEBUG FOR VERBOSE FILE LOGGING VVVVV ---
def setup_logging(log_dir="logs", level=logging.DEBUG):
# --- ^^^^^ SET DEFAULT LEVEL TO DEBUG FOR VERBOSE FILE LOGGING ^^^^^ ---
    """Configures verbose file logging ONLY."""
    try:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(log_dir, f"reconstruction_{timestamp}.log")

        # Configure root logger for file output only
        logging.basicConfig(
            level=level, # Set the root logger level (DEBUG for verbose)
            format='%(asctime)s - %(name)s - %(levelname)-8s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            # --- VVVVV REMOVED StreamHandler VVVVV ---
            handlers=[
                logging.FileHandler(log_filename, mode='w'), # Log ONLY to file
                # logging.StreamHandler() # REMOVED Console output
            ],
            # --- ^^^^^ REMOVED StreamHandler ^^^^^ ---
            force=True # Added force=True to allow reconfiguration if called multiple times
        )

        # Set higher levels for noisy libraries to keep log file cleaner
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING) # Pillow library often used by matplotlib
        # logging.getLogger("cupy").setLevel(logging.INFO) # Keep CuPy INFO for now

        # Log initial messages *to the file*
        # Use the root logger directly here as module loggers might not be set up yet
        logging.info(f"--- Logging Initialized ---")
        logging.info(f"Logging level set to: {logging.getLevelName(level)}")
        logging.info(f"Log file: {log_filename}")
        logging.info(f"-----------------------------")


    except Exception as e:
        # Fallback to basic console logging if file setup fails
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
        logging.critical(f"!!! FAILED TO CONFIGURE FILE LOGGING: {e} !!! Logging to console only.")


# Example usage reminder (no changes needed in other files):
# import logging
# logger = logging.getLogger(__name__)
# logger.debug("This goes only to the file if level=DEBUG")
# logger.info("This goes only to the file if level=INFO or DEBUG")