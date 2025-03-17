'''
This script processes video frames and generates segmentation masks using the SAM2 model. The main steps are as follows:

1. **CUDA Setup**: The script checks if CUDA (GPU support) is available and selects the device accordingly (GPU if available, otherwise CPU).

2. **DeepLabCut (DLC) CSV Extraction**:
   - **read_DLC_csv()**: This function reads the DeepLabCut CSV file containing body part tracking data.
   - It formats the DataFrame by setting the first row as the new column names and creating a MultiIndex with two levels: one for the original column names and one for the body parts (e.g., 'x', 'y', 'likelihood').
   - It also converts each column's data to numeric values, coercing any errors into NaNs.

3. **Frame Extraction from Video**:
   - **create_frames_directory()**: The video is split into individual frames, which are saved into subdirectories (batches) to avoid memory overload.
   - Frames are processed in batches based on the specified batch size.

4. **Coordinate Extraction and Mask Generation**:
   - **extract_coordinate_by_likelihood()**: This function extracts coordinates (x, y) of specific body parts, based on the highest likelihood values. It calculates the average likelihood across all body parts for each frame and selects the frame with the highest likelihood. In case of a tie, it randomly selects one frame.
   - The coordinates of the body parts are returned for further processing.

5. **Segmentation with SAM2**:
   - **segment_object()**: The SAM2 model is used to generate segmentation masks based on the extracted coordinates. The model propagates these masks both forwards and backwards through the video.
   - The script generates binary masks for each frame and stores them in a dictionary.

6. **Saving Masks**:
   - The generated masks are processed to ensure they are in the correct format (binary, 8-bit).
   - The final masks are saved to a TIFF file, sorted by the global frame index.

7. **Cleanup**:
   - **clear_temp_folder()**: Once the processing is complete, the temporary folder containing the extracted frames is deleted.

The final output is a TIFF file containing the binary masks for each video frame, created using the SAM2 model and the body part coordinates from the DLC data.
'''

import torch

# Check if CUDA is available and display the GPU information
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    for i in range(gpu_count):
        print(f"CUDA is available. Using GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Using CPU.")

from PIL import Image
import numpy as np
import pandas as pd
import sys
import os
import argparse
import tifffile as tiff
import cv2
import shutil

def read_DLC_csv(csv_file_path):

    df = pd.read_csv(csv_file_path)

    #remove column names and set first row to new column name
    df.columns = df.iloc[0]
    df = df[1:]

    # Get the first row (which will become the second level of column names)
    second_level_names = df.iloc[0]

    # Create a MultiIndex for columns using the existing column names as the first level
    first_level_names = df.columns
    multi_index = pd.MultiIndex.from_arrays([first_level_names, second_level_names])

    # Set the new MultiIndex as the columns of the DataFrame
    df.columns = multi_index

    # Remove the first row from the DataFrame as it's now used for column names
    df = df.iloc[1:]

    # Removing the first column (index 0)
    df = df.drop(df.columns[0], axis=1)
    df = df.reset_index(drop=True)

    # Convert each column to numeric, coerce errors to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(isinstance(df.columns, pd.MultiIndex))
    print(list(df.columns))

    return df

def extract_coordinate_by_likelihood(df, bodyparts):
        # Step 1: Identify the 'likelihood' columns dynamically
    likelihood_cols = [col for col in df.columns if col[1] == 'likelihood']
    
    # Step 2: Compute the average likelihood per row without modifying the DataFrame
    avg_likelihood = df[likelihood_cols].mean(axis=1)
    
    # Step 3: Find the maximum average likelihood
    max_avg_likelihood = avg_likelihood.max()
    
    # Step 4: Identify the row(s) with the maximum average likelihood
    max_indices = avg_likelihood[avg_likelihood == max_avg_likelihood].index
    
    # Step 5: Randomly select one row if there is a tie
    if len(max_indices) > 1:
        selected_index = np.random.choice(max_indices)
    else:
        selected_index = max_indices[0]

    # Step 6: Retrieve the entire row of the selected entry
    selected_row = df.loc[[selected_index]]

    #extract x and y coordinates as list
    result = {}
    for bodypart in bodyparts:
        if bodypart in selected_row.columns.get_level_values(0):
            x_values = pd.to_numeric(selected_row[bodypart]['x'], errors='coerce')
            y_values = pd.to_numeric(selected_row[bodypart]['y'], errors='coerce')
            result[bodypart] = list(zip(x_values, y_values))
    
    return result, selected_index

def multiply_columns_by_downsample_factor(DLC_data, downsample_factor):
    # Iterate over each column in the DataFrame
    for col in DLC_data.columns:
        # Check if the second level of the column name is 'x' or 'y'
        if col[1] in ['x', 'y']:
            # Multiply the column values by the downsample_factor
            DLC_data[col] *= downsample_factor
    return DLC_data

def create_frames_directory(video_path, batch_size):
    video_dir = os.path.dirname(video_path)
    frames_dir = os.path.join(video_dir, 'frame_directory')
    os.makedirs(frames_dir, exist_ok=True)
    
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        raise ValueError(f"No frames found in video file: {video_path}")
    
    saved_count = 0
    batch_number = 0
    batch_frame_counts = [] #saves batchnumber and frames inside this batchnumber
    current_batch_count = 0
    
    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Create a new batch subfolder when necessary
            if current_batch_count == 0:
                batch_dir = os.path.join(frames_dir, f"{batch_number}")
                os.makedirs(batch_dir, exist_ok=True)
            
            frame_filename = os.path.join(batch_dir, f"{saved_count:06d}.jpg")  # 6 digits for larger video frame counts
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            current_batch_count += 1
            
            if current_batch_count == batch_size:  # Close batch when full
                batch_frame_counts.append((batch_number, current_batch_count))
                current_batch_count = 0
                batch_number += 1
            
            print(f"Saved frame {saved_count}/{total_frames} in folder {batch_number}")
        
        # Add the last batch to the list if it wasn't full
        if current_batch_count > 0:
            batch_frame_counts.append((batch_number, current_batch_count))
    
    finally:
        video.release()
    
    print(f"Extracted {saved_count} frames to {frames_dir} (from total {total_frames} frames)")
    print(f"Organized into {batch_number + 1} folders")
    
    return frames_dir, batch_frame_counts


def check_image_folder(video_path, batch_size):
    """
    Check if folder has enough frames, retry for up to 10 minutes.
    """
    import time

    max_wait_time = 600  # 10 minutes in seconds
    elapsed_time = 0
    wait_interval = 10  # wait 10 seconds between checks

    while elapsed_time < max_wait_time:
        frame_count = len([f for f in os.listdir(video_path) if f.lower().endswith(('.jpg', '.jpeg'))])

        if frame_count >= batch_size:
            return True

        print(f"Found {frame_count}/{batch_size} frames. Waiting 10 seconds... (Total wait: {elapsed_time} seconds)")
        time.sleep(wait_interval)
        elapsed_time += wait_interval

    raise RuntimeError("Failed to generate image folders: Timeout after 10 minutes waiting for frames")

def segment_object(predictor, video_path, coordinate, frame_number, batch_number, batch_size):
    print("\n=== Starting segment_object function ===")
    print(f"Batch Folder Path (video_path): {video_path}")
    print(f"Frame number: {frame_number}")
    print(f"Batch number: {batch_number}")
    print(f"Batch size: {batch_size}")
    print(f"Coordinate keys: {list(coordinate.keys())}")
    print(f"Full coordinate data: {coordinate}")

    # Check if video_path is a non-empty string
    if not isinstance(video_path, str) or not video_path.strip():
        print("ERROR: Provided video_path is invalid or empty.")
    else:
        print("DEBUG: video_path appears valid.")

    # Check if coordinate is a non-empty dict with proper format (keys mapping to list of (x,y) tuples)
    if not isinstance(coordinate, dict) or not coordinate:
        print("ERROR: Provided coordinate is invalid or empty.")
    else:
        for key, points in coordinate.items():
            if not isinstance(points, list) or not all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in points):
                print(f"ERROR: Coordinate for '{key}' is not in the expected format.")
            else:
                print(f"DEBUG: Coordinate for '{key}' is valid with {len(points)} point(s).")

    print(f"\nGenerating masklet for coordinate {coordinate} on frame {frame_number}")

    # model only sees 1 batch as whole video
    normalized_frame_number = frame_number - (batch_number * batch_size)
    print(f"Normalized Frame Number for Batch: {normalized_frame_number}")

    # Initialize lists to hold points and labels
    points_list = []
    labels_list = []

    print("\n=== Processing coordinates ===")
    # Iterate over each key in the coordinate dictionary
    for key in coordinate:
        print(f"Processing bodypart: {key}")
        # Iterate over each (x, y) tuple in the list associated with the key
        for (x, y) in coordinate[key]:
            points_list.append([x, y])
            labels_list.append(1)
            print(f"Added point: ({x}, {y}) for {key}")

    # Convert the lists to NumPy arrays
    import numpy as np  # Ensure numpy is imported
    points = np.array(points_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)

    print("\n=== Points and Labels Information ===")
    print(f"Points array shape: {points.shape}")
    print(f"Points dtype: {points.dtype}")
    print(f"Points:\n{points}")
    print(f"Labels array shape: {labels.shape}")
    print(f"Labels:\n{labels}")

    print("\n=== Checking Image Folder ===")
    '''
    #causes a bug lets leave that for now!
    try:
        check_image_folder(video_path, batch_size)
        print("Image folder check passed successfully")
    except RuntimeError as e:
        print(f"ERROR in image folder check: {str(e)}")
        raise
    '''

    print("\n=== Initializing Inference State ===")
    print(f"DEBUG: Initializing model with video_path: {video_path}")
    try:
        inference_state = predictor.init_state(video_path=video_path)
        print("Inference state initialized successfully")
        # Print the initial inference state for debugging
        print("DEBUG: Initial Inference State:")
        print(inference_state)
    except Exception as e:
        print(f"ERROR initializing inference state: {str(e)}")
        raise

    print("\n=== Adding Points to Model ===")
    try:
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=normalized_frame_number,
            obj_id=0,
            points=points,
            labels=labels,
        )
        print("Points added to model successfully")
        print(f"Model outputs:")
        print(f"out_obj_ids: {out_obj_ids}")
        print(f"out_mask_logits shape: {out_mask_logits.shape}")
        print(f"out_mask_logits dtype: {out_mask_logits.dtype}")
        print(f"out_mask_logits min/max: {out_mask_logits.min():.4f}/{out_mask_logits.max():.4f}")
        print(f"out_mask_logits mean/std: {out_mask_logits.mean():.4f}/{out_mask_logits.std():.4f}")
    except Exception as e:
        print(f"ERROR adding points to model: {str(e)}")
        raise

    # Dictionary to store masks for video frames
    video_segments = {}
    print("\n=== Starting Forward Propagation ===")

    # Forward propagation (reverse=False)
    mask_count = 0
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=False):
        print(f"\nFrame {out_frame_idx} model outputs:")
        print(f"out_obj_ids: {out_obj_ids}")
        print(f"out_mask_logits shape: {out_mask_logits.shape}")
        print(f"out_mask_logits dtype: {out_mask_logits.dtype}")
        print(f"out_mask_logits min/max: {out_mask_logits.min():.4f}/{out_mask_logits.max():.4f}")
        print(f"out_mask_logits mean/std: {out_mask_logits.mean():.4f}/{out_mask_logits.std():.4f}")

        for i, out_obj_id in enumerate(out_obj_ids):
            # Generate mask from logits
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()
            mask = np.squeeze(mask)

            print(f"\nProcessing frame {out_frame_idx}")
            print(f"Mask shape: {mask.shape}")

            if len(mask.shape) == 2:
                binary_mask = (mask * 255).astype(np.uint8)
                unique_vals = np.unique(binary_mask)
                print(f"Binary mask unique values: {unique_vals}")
                print(f"Binary mask shape: {binary_mask.shape}")

                video_segments[out_frame_idx] = binary_mask
                mask_count += 1
            else:
                print(f"WARNING: Non-2D mask detected for frame {out_frame_idx}")
                black_mask = np.zeros(mask.shape[1:], dtype=np.uint8)
                video_segments[out_frame_idx] = black_mask

    print(f"\nForward propagation complete. Generated {mask_count} masks.")
    print("\n=== Starting Reverse Propagation ===")

    # Reverse propagation code (similar debug statements...)
    mask_count = 0
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
        print(f"\nFrame {out_frame_idx} model outputs (reverse):")
        print(f"out_obj_ids: {out_obj_ids}")
        print(f"out_mask_logits shape: {out_mask_logits.shape}")
        print(f"out_mask_logits dtype: {out_mask_logits.dtype}")
        print(f"out_mask_logits min/max: {out_mask_logits.min():.4f}/{out_mask_logits.max():.4f}")
        print(f"out_mask_logits mean/std: {out_mask_logits.mean():.4f}/{out_mask_logits.std():.4f}")

        for i, out_obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()
            mask = np.squeeze(mask)

            print(f"\nProcessing frame {out_frame_idx} (reverse)")
            print(f"Mask shape: {mask.shape}")

            if len(mask.shape) == 2:
                binary_mask = (mask * 255).astype(np.uint8)
                unique_vals = np.unique(binary_mask)
                print(f"Binary mask unique values: {unique_vals}")
                print(f"Binary mask shape: {binary_mask.shape}")

                video_segments[out_frame_idx] = binary_mask
                mask_count += 1
            else:
                print(f"WARNING: Non-2D mask detected for frame {out_frame_idx}")
                black_mask = np.zeros(mask.shape[1:], dtype=np.uint8)
                video_segments[out_frame_idx] = black_mask

    print(f"\nReverse propagation complete. Generated {mask_count} masks.")
    print(f"Total frames in video_segments: {len(video_segments)}")

    # Sort and return
    return dict(sorted(video_segments.items()))


def process_mask(mask):
    """
    Process the input mask to ensure it's in the proper format:
    8-bit, single-channel binary image.
    
    Args:
    mask (numpy.ndarray): Input mask image
    
    Returns:
    numpy.ndarray: Processed binary mask
    """
    # Check the number of channels
    if len(mask.shape) == 2:
        # If it's already single channel
        mask_single_channel = mask
    elif len(mask.shape) == 3:
        if mask.shape[2] == 3:
            # If it's a 3-channel image, convert to grayscale
            mask_single_channel = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        elif mask.shape[2] == 4:
            # If it's a 4-channel image (with alpha), convert to grayscale
            mask_single_channel = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"Unexpected number of channels: {mask.shape[2]}")
    else:
        raise ValueError(f"Unexpected shape of mask: {mask.shape}")

    # Ensure 8-bit depth
    mask_8bit = cv2.convertScaleAbs(mask_single_channel)

    # Apply threshold to create binary mask
    _, mask_binary = cv2.threshold(mask_8bit, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return mask_binary


def clear_temp_folder(frames_dir):
    """
    Deletes the temporary folder containing the batch frames after processing is complete.

    Args:
    frames_dir (str): The path to the directory containing batch frames.
    """
    try:
        # Remove the entire directory tree at frames_dir
        shutil.rmtree(frames_dir)
        print(f"Temporary folder '{frames_dir}' has been deleted successfully.")
    except Exception as e:
        print(f"Error deleting temporary folder '{frames_dir}': {str(e)}")

    
def main(args):

    # Add deterministic controls to ensure reproducible results
    torch.manual_seed(42)  # Set the seed for PyTorch's CPU random number generator

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)  # Set the same seed for all GPU devices

    # Force PyTorch to use deterministic algorithms
    # This might slow down performance but ensures reproducibility
    torch.backends.cudnn.deterministic = True

    # Disable automatic tuning of cuDNN convolution algorithms
    # This prevents cuDNN from selecting fastest but potentially non-deterministic algorithms
    torch.backends.cudnn.benchmark = False

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process input files and generate output.")
    parser.add_argument("-video_path", type=str, help="Path to the input file")
    parser.add_argument("-output_file_path", type=str, help="Path to the output file")
    parser.add_argument("-DLC_csv_file_path", type=str, help="Path to the DLC CSV file")
    parser.add_argument("-column_names", type=str, nargs='+', help="List of column names")
    parser.add_argument("-SAM2_path", type=str, help="Location of GitRepo!")
    parser.add_argument("--downsample_factor", type=float, default=0, help="Use in case DLC was used on downsampled video (default: 0)")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for processing (default: 100)")

    # Parse the arguments
    args = parser.parse_args(args)
    
    print("Starting SAM2 Video Processing")
    
    SAM2_path = args.SAM2_path

    # Define paths relative to the base path
    checkpoint = os.path.join(SAM2_path , "checkpoints", "sam2_hiera_large.pt")
    model_cfg = "/" + os.path.join(SAM2_path, "sam2_configs", "sam2_hiera_l.yaml")

    # Check if files exist
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
    if not os.path.exists(model_cfg):
        raise FileNotFoundError(f"Config file not found: {model_cfg}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    from sam2.build_sam import build_sam2_video_predictor

    print("Loading SAM2 model")
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    
    # Access the parsed arguments
    video_path = args.video_path
    output_file_path = args.output_file_path
    DLC_csv_file_path = args.DLC_csv_file_path
    column_names = args.column_names
    downsample_factor = args.downsample_factor
    batch_size = args.batch_size

    print(f"Video File: {video_path}")
    print(f"Output file: {output_file_path}")
    print(f"DLC CSV file: {DLC_csv_file_path}")
    print(f"Column names: {column_names}")
    print(f"Batchsize: {batch_size}")

    print("Extracting frames from video")
    frames_dir, batch_frame_count = create_frames_directory(video_path, batch_size)

    DLC_data = read_DLC_csv(DLC_csv_file_path)

    if downsample_factor != 0:
        DLC_data = multiply_columns_by_downsample_factor(DLC_data, downsample_factor)
    

    # Initialize a variable to hold the final concatenated mask
    final_mask_dict = {}

    for batch_number, _ in batch_frame_count:
        # Directory for the current batch of frames
        batch_dir = os.path.join(frames_dir, f"{batch_number}")

        # Slice the DataFrame for the current batch
        DLC_data_batch = DLC_data.iloc[batch_number * batch_size : (batch_number + 1) * batch_size]

        # Extract coordinates and frame numbers for the current batch
        coordinates, frame_number = extract_coordinate_by_likelihood(DLC_data_batch, column_names)

        # Generate mask for the current batch of frames
        print(f"Generating masklet for batch {batch_number}")

        masks = segment_object(predictor, batch_dir, coordinates, frame_number, batch_number, batch_size)
        print(f"Masklet generated for batch {batch_number}. Number of masks: {len(masks)}. Stored masks: {list(masks.keys())}")

         # Concatenate masks into the final dictionary with global frame indexing
        for out_frame_idx, binary_mask in masks.items():
            global_frame_idx = out_frame_idx + (batch_number * batch_size)  # Adjust frame index to global frame count
            final_mask_dict[global_frame_idx] = binary_mask  # Store or update the mask for the global frame

    """
    Process and save the final mask dictionary into a TIFF file.
    """
    # Sort th
    # e dictionary by the global frame index (just in case)
    sorted_mask_keys = sorted(final_mask_dict.keys())

    print(f"Total number of masks in final_mask_dict: {len(sorted_mask_keys)}")
    print(f"Frame indices covered: from {min(sorted_mask_keys)} to {max(sorted_mask_keys)}")

    with tiff.TiffWriter(output_file_path, bigtiff=True) as tif_writer:
        for global_frame_idx in sorted_mask_keys:
            mask = final_mask_dict[global_frame_idx]
            processed_mask = process_mask(mask)
            # Just write the mask, whether it's black or not
            tif_writer.write(processed_mask, contiguous=True)
    
    print(f"All masks saved to {output_file_path}.")

    # After everything is finished, delete the temporary folder
    clear_temp_folder(frames_dir)

if __name__ == "__main__":
     main(sys.argv[1:])  # exclude the script name from the args when called from sh
