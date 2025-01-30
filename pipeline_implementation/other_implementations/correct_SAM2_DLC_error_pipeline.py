import cv2
import numpy as np
import argparse
import sys
import logging
import dask.array as da
import tifffile as tiff
from imutils import MicroscopeDataReader
from tqdm import tqdm

def process_binary_mask(mask, lower_threshold, upper_threshold):
    # Find all objects in the mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels == 1:  # Only background
        return np.zeros_like(mask)
    
    # Get the sizes of all objects (excluding background)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    
    # Check if the largest object is between the thresholds
    largest_size = np.max(sizes)
    if lower_threshold <= largest_size <= upper_threshold:
        return mask
    
    # If not, invert the mask
    inverted_mask = cv2.bitwise_not(mask)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted_mask, connectivity=8)
    
    if num_labels == 1:  # Only background
        return np.zeros_like(mask)
    
    sizes = stats[1:, cv2.CC_STAT_AREA]
    largest_size = np.max(sizes)
    
    # Check if the largest object in the inverted mask is between the thresholds
    if lower_threshold <= largest_size <= upper_threshold:
        return inverted_mask
    
    # If not, search for any object that fits the criteria
    valid_objects = np.logical_and(lower_threshold <= sizes, sizes <= upper_threshold)
    
    if np.any(valid_objects):
        result_mask = np.zeros_like(mask)
        for i, is_valid in enumerate(valid_objects, 1):
            if is_valid:
                result_mask = cv2.bitwise_or(result_mask, (labels == i).astype(np.uint8))
        return result_mask
    
    # If still nothing fits, return a black mask
    return np.zeros_like(mask)

def main(args):
    parser = argparse.ArgumentParser(description="Process a stack of binary masks based on object size thresholds.")
    parser.add_argument("input_file_path", help="Path to the input binary mask stack")
    parser.add_argument("output_file_path", help="Path to save the output mask stack")
    parser.add_argument("lower_threshold", type=int, help="Lower size threshold")
    parser.add_argument("upper_threshold", type=int, help="Upper size threshold")
    
    args = parser.parse_args(args)

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        reader_obj = MicroscopeDataReader(args.input_file_path, as_raw_tiff=True, raw_tiff_num_slices=1)
        tif = da.squeeze(reader_obj.dask_array)

        with tiff.TiffWriter(args.output_file_path, bigtiff=True) as tif_writer:
            total_frames = len(tif)
            
            # Wrap the processing loop with tqdm for a progress bar
            for i, mask in tqdm(enumerate(tif), total=total_frames, desc="Processing masks"):
                mask = np.array(mask)

                # Ensure the mask is binary
                _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

                # Process the mask
                result_mask = process_binary_mask(binary_mask, args.lower_threshold, args.upper_threshold)

                tif_writer.write(result_mask, contiguous=True)

        logging.info(f"All masks processed. Output saved to {args.output_file_path}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    logging.info(f"Shell commands passed: {sys.argv}")
    main(sys.argv[1:])  # exclude the script name from the args when called from shell
