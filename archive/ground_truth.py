# In the loading section
ground_truth = None
try:
    with h5py.File(input_filename, 'r') as f:
        if 'ground_truth' in f:
            ground_truth = f['ground_truth'][:]
except Exception as e:
    print(f"Error loading ground truth: {e}")

# If ground truth exists, use it; otherwise, regenerate
if ground_truth is not None:
    gt_baseband_compare = ground_truth[:num_samples_gt_compare]
else:
    # [Your existing regeneration code]
    pass