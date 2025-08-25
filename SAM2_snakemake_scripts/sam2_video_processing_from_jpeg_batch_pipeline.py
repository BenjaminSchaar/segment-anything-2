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
import gc

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
from sam2.utils.lazy_video_provider import create_lazy_video_provider

def print_gpu_memory(prefix=""):
    """
    Print current GPU memory usage with GB precision.
    
    Args:
    prefix (str): Descriptive prefix for the memory report
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert bytes to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)   # Convert bytes to GB
        print(f"{prefix}GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    else:
        print(f"{prefix}CUDA not available - no GPU memory to report")

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

def segment_object(predictor, video_path, coordinate, frame_number, batch_number, batch_size):

    print("Batch Folder Path", video_path)

    print(f"Generating masklet for coordinate {coordinate} on frame {frame_number}")

    #model only sees 1 batch as whole video
    normalized_frame_number = frame_number - (batch_number * batch_size)
    print("Normalized Framenumber for Batch:", normalized_frame_number)

     # Initialize lists to hold points and labels
    points_list = []
    labels_list = []
    
    # Iterate over each key (e.g., 'vulva', 'neck') in the coordinate dictionary
    for key in coordinate:
        # Iterate over each (x, y) tuple in the list associated with the key
        for (x, y) in coordinate[key]:
            points_list.append([x, y])
            labels_list.append(1)  # You can customize the label as needed

    # Convert the lists to NumPy arrays
    points = np.array(points_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)
    
    # Now you can use 'points' and 'labels' as needed
    print("Points array:", points)
    print("Labels array:", labels)

    # Initialize inference state
    inference_state = predictor.init_state(video_path=video_path)
    print("Initialized inference state")

    # Add points to model and get mask logits
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=normalized_frame_number,
        obj_id=0,
        points=points,
        labels=labels,
    )
    
    print("Added point to the model")

    # Dictionary to store masks for video frames
    video_segments = {}
    print("Propagating masklet through video frames (forward direction):")

    # Forward propagation (reverse=False)
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=False):
        for i, out_obj_id in enumerate(out_obj_ids):
            # Generate mask from logits
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()

            # Remove the extra channel dimension if present
            mask = np.squeeze(mask)
            
            # Check if the mask is 2D (binary mask) and proceed
            if len(mask.shape) == 2:
                # Convert mask to binary (0 or 255)
                binary_mask = (mask * 255).astype(np.uint8)
                
                # Show unique values and shape of the binary mask
                print(f"Binary mask for frame {out_frame_idx} has unique values: {np.unique(binary_mask)} and shape: {binary_mask.shape}")
                
                # Add the binary mask to video_segments (no second key, just out_frame_idx)
                video_segments[out_frame_idx] = binary_mask
                print(f"Processed frame {out_frame_idx}")
            else:
                # Create a black mask (all zeros) with the same width and height as the original mask
                black_mask = np.zeros(mask.shape[1:], dtype=np.uint8)
                
                # Add the black mask to video_segments (no second key)
                video_segments[out_frame_idx] = black_mask
                print(f"Non-2D mask for frame {out_frame_idx}. Added black mask instead.")

    # Reverse propagation (reverse=True)
    print("Propagating masklet through video frames (reverse direction):")
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
        for i, out_obj_id in enumerate(out_obj_ids):
            # Generate mask from logits
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()

            # Remove the extra channel dimension if present
            mask = np.squeeze(mask)
            
            # Check if the mask is 2D (binary mask) and proceed
            if len(mask.shape) == 2:
                # Convert mask to binary (0 or 255)
                binary_mask = (mask * 255).astype(np.uint8)
                
                # Show unique values and shape of the binary mask
                print(f"Binary mask for frame {out_frame_idx} has unique values: {np.unique(binary_mask)} and shape: {binary_mask.shape}")
                
                # Add the binary mask to video_segments (no second key)
                video_segments[out_frame_idx] = binary_mask
                print(f"Processed frame {out_frame_idx}")
            else:
                # Create a black mask (all zeros) with the same width and height as the original mask
                black_mask = np.zeros(mask.shape[1:], dtype=np.uint8)
                
                # Add the black mask to video_segments (no second key)
                video_segments[out_frame_idx] = black_mask
                print(f"Non-2D mask for frame {out_frame_idx}. Added black mask instead.")

    # Sort the dictionary to ensure the frames are in proper order
    video_segments = dict(sorted(video_segments.items()))

    return video_segments

def segment_object_from_arrays(predictor, image_array, coordinate, frame_number, batch_number, batch_size):
    """
    Generate segmentation masks from image arrays using SAM2.
    
    Args:
    predictor: SAM2 video predictor instance
    image_array: numpy array of shape (num_frames, height, width, 3)
    coordinate: dict of body part coordinates
    frame_number: frame index within the batch
    batch_number: current batch number
    batch_size: batch size for processing
    
    Returns:
    dict: Dictionary mapping frame indices to binary masks
    """
    
    print(f"Generating masklet for coordinate {coordinate} on frame {frame_number}")
    print(f"Image array shape: {image_array.shape}")

    # Initialize lists to hold points and labels
    points_list = []
    labels_list = []
    
    # Iterate over each key (e.g., 'vulva', 'neck') in the coordinate dictionary
    for key in coordinate:
        # Iterate over each (x, y) tuple in the list associated with the key
        for (x, y) in coordinate[key]:
            points_list.append([x, y])
            labels_list.append(1)  # You can customize the label as needed

    # Convert the lists to NumPy arrays
    points = np.array(points_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)
    
    # Now you can use 'points' and 'labels' as needed
    print("Points array:", points)
    print("Labels array:", labels)

    # Initialize inference state with image array instead of video path
    from sam2.utils._load_images_function_from_array import load_video_frames_from_array
    
    # Prepare the image array for SAM2 (normalize from 0-255 to 0-1)
    image_array_normalized = image_array.astype(np.float32) / 255.0
    
    # Load frames using the array-based loader
    images, video_height, video_width = load_video_frames_from_array(
        image_array=image_array_normalized,
        image_size=predictor.image_size,
        offload_video_to_cpu=False,
        compute_device=predictor.device
    )
    
    # Create inference state manually (similar to init_state but with arrays)
    inference_state = {}
    inference_state["images"] = images
    inference_state["num_frames"] = len(images)
    inference_state["offload_video_to_cpu"] = False
    inference_state["offload_state_to_cpu"] = False
    inference_state["video_height"] = video_height
    inference_state["video_width"] = video_width
    inference_state["device"] = predictor.device
    inference_state["storage_device"] = predictor.device
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    inference_state["cached_features"] = {}
    inference_state["constants"] = {}
    from collections import OrderedDict
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    inference_state["output_dict"] = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }
    inference_state["output_dict_per_obj"] = {}
    inference_state["temp_output_dict_per_obj"] = {}
    inference_state["consolidated_frame_inds"] = {
        "cond_frame_outputs": set(),
        "non_cond_frame_outputs": set(),
    }
    inference_state["tracking_has_started"] = False
    inference_state["frames_already_tracked"] = {}
    
    print("Initialized inference state with image arrays")

    # Add points to model and get mask logits
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_number,
        obj_id=0,
        points=points,
        labels=labels,
    )
    
    print("Added point to the model")

    # Dictionary to store masks for video frames
    video_segments = {}
    print("Propagating masklet through video frames (forward direction):")

    # Forward propagation (reverse=False)
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=False):
        for i, out_obj_id in enumerate(out_obj_ids):
            # Generate mask from logits
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()

            # Remove the extra channel dimension if present
            mask = np.squeeze(mask)
            
            # Check if the mask is 2D (binary mask) and proceed
            if len(mask.shape) == 2:
                # Convert mask to binary (0 or 255)
                binary_mask = (mask * 255).astype(np.uint8)
                
                # Show unique values and shape of the binary mask
                print(f"Binary mask for frame {out_frame_idx} has unique values: {np.unique(binary_mask)} and shape: {binary_mask.shape}")
                
                # Add the binary mask to video_segments (no second key, just out_frame_idx)
                video_segments[out_frame_idx] = binary_mask
                print(f"Processed frame {out_frame_idx}")
            else:
                # Create a black mask (all zeros) with the same width and height as the original mask
                black_mask = np.zeros(mask.shape[1:], dtype=np.uint8)
                
                # Add the black mask to video_segments (no second key)
                video_segments[out_frame_idx] = black_mask
                print(f"Non-2D mask for frame {out_frame_idx}. Added black mask instead.")

    # Reverse propagation (reverse=True)
    print("Propagating masklet through video frames (reverse direction):")
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
        for i, out_obj_id in enumerate(out_obj_ids):
            # Generate mask from logits
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()

            # Remove the extra channel dimension if present
            mask = np.squeeze(mask)
            
            # Check if the mask is 2D (binary mask) and proceed
            if len(mask.shape) == 2:
                # Convert mask to binary (0 or 255)
                binary_mask = (mask * 255).astype(np.uint8)
                
                # Show unique values and shape of the binary mask
                print(f"Binary mask for frame {out_frame_idx} has unique values: {np.unique(binary_mask)} and shape: {binary_mask.shape}")
                
                # Add the binary mask to video_segments (no second key)
                video_segments[out_frame_idx] = binary_mask
                print(f"Processed frame {out_frame_idx}")
            else:
                # Create a black mask (all zeros) with the same width and height as the original mask
                black_mask = np.zeros(mask.shape[1:], dtype=np.uint8)
                
                # Add the black mask to video_segments (no second key)
                video_segments[out_frame_idx] = black_mask
                print(f"Non-2D mask for frame {out_frame_idx}. Added black mask instead.")

    # Sort the dictionary to ensure the frames are in proper order
    video_segments = dict(sorted(video_segments.items()))

    # Cleanup intermediate variables and clear GPU cache before return
    del image_array_normalized
    del images
    torch.cuda.empty_cache()

    return video_segments

def segment_object_with_lazy_video(predictor, video_path, coordinate, frame_number, batch_number, batch_size):
    """
    Generate segmentation masks using lazy video loading for memory efficiency.
    
    Args:
    predictor: SAM2 video predictor instance
    video_path: Path to the video file
    coordinate: dict of body part coordinates
    frame_number: frame index within the batch
    batch_number: current batch number
    batch_size: batch size for processing
    
    Returns:
    dict: Dictionary mapping frame indices to binary masks
    """
    
    print(f"Generating masklet for coordinate {coordinate} on frame {frame_number} using lazy loading")
    print(f"Processing video: {video_path}")

    # Initialize lists to hold points and labels
    points_list = []
    labels_list = []
    
    # Iterate over each key (e.g., 'vulva', 'neck') in the coordinate dictionary
    for key in coordinate:
        # Iterate over each (x, y) tuple in the list associated with the key
        for (x, y) in coordinate[key]:
            points_list.append([x, y])
            labels_list.append(1)  # You can customize the label as needed

    # Convert the lists to NumPy arrays
    points = np.array(points_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)
    
    # Now you can use 'points' and 'labels' as needed
    print("Points array:", points)
    print("Labels array:", labels)

    # Create lazy video provider instead of loading all frames
    batch_start_frame = batch_number * batch_size
    lazy_provider = create_lazy_video_provider(
        video_path=video_path,
        batch_start_frame=batch_start_frame,
        batch_size=batch_size,
        image_size=predictor.image_size,
        device=predictor.device,
        cache_size=50,  # Keep up to 50 frames in memory
        prefetch_size=5  # Prefetch 5 frames ahead
    )
    
    try:
        # Create inference state with lazy provider
        inference_state = {}
        inference_state["images"] = lazy_provider  # SAM2 will call lazy_provider[frame_idx] 
        inference_state["num_frames"] = len(lazy_provider)
        inference_state["offload_video_to_cpu"] = False
        inference_state["offload_state_to_cpu"] = False
        inference_state["video_height"] = lazy_provider.video_height or 896  # Default if not loaded yet
        inference_state["video_width"] = lazy_provider.video_width or 900    # Default if not loaded yet
        inference_state["device"] = predictor.device
        inference_state["storage_device"] = predictor.device
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        inference_state["cached_features"] = {}
        inference_state["constants"] = {}
        from collections import OrderedDict
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        inference_state["output_dict_per_obj"] = {}
        inference_state["temp_output_dict_per_obj"] = {}
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),
            "non_cond_frame_outputs": set(),
        }
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        
        print("Initialized inference state with lazy video provider")

        # Add points to model and get mask logits
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_number,
            obj_id=0,
            points=points,
            labels=labels,
        )
        
        print("Added point to the model")

        # Dictionary to store masks for video frames
        video_segments = {}
        print("Propagating masklet through video frames (forward direction):")

        # Forward propagation (reverse=False)
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=False):
            for i, out_obj_id in enumerate(out_obj_ids):
                # Generate mask from logits
                mask = (out_mask_logits[i] > 0.0).cpu().numpy()

                # Remove the extra channel dimension if present
                mask = np.squeeze(mask)
                
                # Check if the mask is 2D (binary mask) and proceed
                if len(mask.shape) == 2:
                    # Convert mask to binary (0 or 255)
                    binary_mask = (mask * 255).astype(np.uint8)
                    
                    # Show unique values and shape of the binary mask
                    print(f"Binary mask for frame {out_frame_idx} has unique values: {np.unique(binary_mask)} and shape: {binary_mask.shape}")
                    
                    # Add the binary mask to video_segments (no second key, just out_frame_idx)
                    video_segments[out_frame_idx] = binary_mask
                    print(f"Processed frame {out_frame_idx}")
                else:
                    # Create a black mask (all zeros) with the same width and height as the original mask
                    black_mask = np.zeros(mask.shape[1:], dtype=np.uint8)
                    
                    # Add the black mask to video_segments (no second key)
                    video_segments[out_frame_idx] = black_mask
                    print(f"Non-2D mask for frame {out_frame_idx}. Added black mask instead.")

        # Reverse propagation (reverse=True)
        print("Propagating masklet through video frames (reverse direction):")
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
            for i, out_obj_id in enumerate(out_obj_ids):
                # Generate mask from logits
                mask = (out_mask_logits[i] > 0.0).cpu().numpy()

                # Remove the extra channel dimension if present
                mask = np.squeeze(mask)
                
                # Check if the mask is 2D (binary mask) and proceed
                if len(mask.shape) == 2:
                    # Convert mask to binary (0 or 255)
                    binary_mask = (mask * 255).astype(np.uint8)
                    
                    # Show unique values and shape of the binary mask
                    print(f"Binary mask for frame {out_frame_idx} has unique values: {np.unique(binary_mask)} and shape: {binary_mask.shape}")
                    
                    # Add the binary mask to video_segments (no second key)
                    video_segments[out_frame_idx] = binary_mask
                    print(f"Processed frame {out_frame_idx}")
                else:
                    # Create a black mask (all zeros) with the same width and height as the original mask
                    black_mask = np.zeros(mask.shape[1:], dtype=np.uint8)
                    
                    # Add the black mask to video_segments (no second key)
                    video_segments[out_frame_idx] = black_mask
                    print(f"Non-2D mask for frame {out_frame_idx}. Added black mask instead.")

        # Sort the dictionary to ensure the frames are in proper order
        video_segments = dict(sorted(video_segments.items()))

        # Print cache statistics for monitoring
        stats = lazy_provider.get_stats()
        print(f"Lazy provider stats: {stats}")

        return video_segments
    
    finally:
        # Cleanup lazy provider cache
        lazy_provider.clear_cache()
        torch.cuda.empty_cache()

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


def load_video_frames_to_array(video_path, batch_size):
    """
    Load video frames directly into numpy arrays organized by batches.
    
    Args:
    video_path (str): Path to the video file
    batch_size (int): Number of frames per batch
    
    Returns:
    tuple: (batch_frame_arrays, batch_frame_counts)
        - batch_frame_arrays: List of numpy arrays, each containing frames for a batch
        - batch_frame_counts: List of tuples (batch_number, frame_count)
    """
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        raise ValueError(f"No frames found in video file: {video_path}")
    
    batch_frame_arrays = []
    batch_frame_counts = []
    current_batch_frames = []
    batch_number = 0
    frame_count = 0
    
    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Convert BGR to RGB for consistency with PIL/SAM2 expectations
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_batch_frames.append(frame_rgb)
            frame_count += 1
            
            if len(current_batch_frames) == batch_size:
                # Convert list of frames to numpy array (batch_size, height, width, 3)
                batch_array = np.stack(current_batch_frames, axis=0)
                batch_frame_arrays.append(batch_array)
                batch_frame_counts.append((batch_number, len(current_batch_frames)))
                
                print(f"Loaded batch {batch_number} with {len(current_batch_frames)} frames")
                current_batch_frames = []
                batch_number += 1
        
        # Handle remaining frames in the last batch
        if current_batch_frames:
            batch_array = np.stack(current_batch_frames, axis=0)
            batch_frame_arrays.append(batch_array)
            batch_frame_counts.append((batch_number, len(current_batch_frames)))
            print(f"Loaded final batch {batch_number} with {len(current_batch_frames)} frames")
    
    finally:
        video.release()
    
    print(f"Loaded {frame_count} frames into {len(batch_frame_arrays)} batches")
    
    return batch_frame_arrays, batch_frame_counts

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
    parser.add_argument("--device", type=str, help="Pass the cluster environment string according to lisc docu")
    parser.add_argument("--use_image_arrays", action="store_true", default=False, help="Use direct image array processing instead of extracting frames to disk (default: False)")
    parser.add_argument("--use_lazy_video", action="store_true", default=False, help="Use lazy video loading for memory-efficient 400-frame processing (default: False)")
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

    if args.device:
        device = torch.device(f"cuda:{args.device}")
    else:
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
    use_image_arrays = args.use_image_arrays
    use_lazy_video = args.use_lazy_video

    print(f"Video File: {video_path}")
    print(f"Output file: {output_file_path}")
    print(f"DLC CSV file: {DLC_csv_file_path}")
    print(f"Column names: {column_names}")
    print(f"Batchsize: {batch_size}")
    print(f"Use image arrays: {use_image_arrays}")
    print(f"Use lazy video loading: {use_lazy_video}")

    # Read DLC data
    DLC_data = read_DLC_csv(DLC_csv_file_path)

    if downsample_factor != 0:
        DLC_data = multiply_columns_by_downsample_factor(DLC_data, downsample_factor)
    

    # Initialize a variable to hold the final concatenated mask
    final_mask_dict = {}

    # Choose processing method based on flag
    if use_lazy_video:
        print("Using lazy video loading for memory-efficient processing")
        # Calculate batch information for lazy loading
        video_cap = cv2.VideoCapture(video_path)
        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_cap.release()
        
        batch_frame_count = []
        for batch_num in range(0, total_frames, batch_size):
            frames_in_batch = min(batch_size, total_frames - batch_num)
            batch_frame_count.append((batch_num // batch_size, frames_in_batch))
        
        frames_dir = None  # No directory needed for lazy loading
        
        for batch_number, frames_in_batch in batch_frame_count:
            # GPU memory monitoring - batch start
            print_gpu_memory(f"[Lazy Batch {batch_number}] Start - ")
            
            try:
                # Slice the DataFrame for the current batch
                DLC_data_batch = DLC_data.iloc[batch_number * batch_size : (batch_number + 1) * batch_size]

                # Extract coordinates and frame numbers for the current batch  
                coordinates, frame_number = extract_coordinate_by_likelihood(DLC_data_batch, column_names)

                # Generate mask for the current batch using lazy video loading
                print(f"Generating masklet for lazy batch {batch_number} with {frames_in_batch} frames")

                masks = segment_object_with_lazy_video(predictor, video_path, coordinates, frame_number, batch_number, batch_size)
                print(f"Masklet generated for lazy batch {batch_number}. Number of masks: {len(masks)}. Stored masks: {list(masks.keys())}")

                # Concatenate masks into the final dictionary with global frame indexing
                for out_frame_idx, binary_mask in masks.items():
                    global_frame_idx = out_frame_idx + (batch_number * batch_size)  # Adjust frame index to global frame count
                    final_mask_dict[global_frame_idx] = binary_mask  # Store or update the mask for the global frame

            except torch.cuda.OutOfMemoryError as e:
                print(f"[Lazy Batch {batch_number}] CUDA OOM Error encountered: {e}")
                # Emergency memory cleanup
                torch.cuda.empty_cache()
                gc.collect()
                print_gpu_memory(f"[Lazy Batch {batch_number}] After emergency cleanup - ")
                # For lazy loading, this shouldn't happen often, but skip if it does
                print(f"[Lazy Batch {batch_number}] Skipping batch due to persistent memory issues")
                continue
            
            finally:
                # Cleanup intermediate variables
                if 'masks' in locals():
                    del masks
                if 'coordinates' in locals():
                    del coordinates
                if 'DLC_data_batch' in locals():
                    del DLC_data_batch
                
                # Force GPU cache clearing after batch completion
                torch.cuda.empty_cache()
                gc.collect()
                
                # GPU memory monitoring - batch end
                print_gpu_memory(f"[Lazy Batch {batch_number}] End - ")
                
    elif use_image_arrays:
        print("Using direct image array processing")
        # Load video frames directly into arrays
        batch_frame_arrays, batch_frame_count = load_video_frames_to_array(video_path, batch_size)
        frames_dir = None  # No directory needed for array processing
        
        for (batch_number, _), batch_array in zip(batch_frame_count, batch_frame_arrays):
            # GPU memory monitoring - batch start
            print_gpu_memory(f"[Batch {batch_number}] Start - ")
            
            try:
                # Slice the DataFrame for the current batch
                DLC_data_batch = DLC_data.iloc[batch_number * batch_size : (batch_number + 1) * batch_size]

                # Extract coordinates and frame numbers for the current batch
                coordinates, frame_number = extract_coordinate_by_likelihood(DLC_data_batch, column_names)

                # Generate mask for the current batch of frames using image arrays
                print(f"Generating masklet for batch {batch_number} with {len(batch_array)} frames")

                masks = segment_object_from_arrays(predictor, batch_array, coordinates, frame_number, batch_number, batch_size)
                print(f"Masklet generated for batch {batch_number}. Number of masks: {len(masks)}. Stored masks: {list(masks.keys())}")

                # Concatenate masks into the final dictionary with global frame indexing
                for out_frame_idx, binary_mask in masks.items():
                    global_frame_idx = out_frame_idx + (batch_number * batch_size)  # Adjust frame index to global frame count
                    final_mask_dict[global_frame_idx] = binary_mask  # Store or update the mask for the global frame

            except torch.cuda.OutOfMemoryError as e:
                print(f"[Batch {batch_number}] CUDA OOM Error encountered: {e}")
                # Emergency memory cleanup
                torch.cuda.empty_cache()
                gc.collect()
                print_gpu_memory(f"[Batch {batch_number}] After emergency cleanup - ")
                
                # Graduated recovery strategy
                original_batch_size = len(batch_array)
                if original_batch_size > 50:  # Try smaller batch size
                    smaller_batch_size = max(25, original_batch_size // 2)
                    print(f"[Batch {batch_number}] Attempting recovery with smaller batch size: {smaller_batch_size}")
                    
                    try:
                        # Split the batch and process in smaller chunks
                        masks = {}
                        for chunk_start in range(0, original_batch_size, smaller_batch_size):
                            chunk_end = min(chunk_start + smaller_batch_size, original_batch_size)
                            chunk_array = batch_array[chunk_start:chunk_end]
                            
                            print(f"[Batch {batch_number}] Processing chunk {chunk_start}-{chunk_end}")
                            chunk_masks = segment_object_from_arrays(predictor, chunk_array, coordinates, 
                                                                   frame_number - chunk_start if frame_number >= chunk_start else 0, 
                                                                   batch_number, smaller_batch_size)
                            
                            # Adjust frame indices for the chunk
                            for chunk_frame_idx, chunk_mask in chunk_masks.items():
                                adjusted_frame_idx = chunk_frame_idx + chunk_start
                                masks[adjusted_frame_idx] = chunk_mask
                            
                            # Cleanup after each chunk
                            del chunk_array
                            del chunk_masks
                            torch.cuda.empty_cache()
                            gc.collect()
                        
                        print(f"[Batch {batch_number}] Recovery successful with chunked processing")
                        # Continue with the recovered masks - store in final_mask_dict
                        for out_frame_idx, binary_mask in masks.items():
                            global_frame_idx = out_frame_idx + (batch_number * batch_size)
                            final_mask_dict[global_frame_idx] = binary_mask
                        
                    except torch.cuda.OutOfMemoryError as recovery_error:
                        print(f"[Batch {batch_number}] Recovery attempt also failed: {recovery_error}")
                        print(f"[Batch {batch_number}] Skipping batch due to persistent memory issues")
                        # Continue with next batch rather than crashing
                        continue
                else:
                    print(f"[Batch {batch_number}] Batch size already small ({original_batch_size}), cannot recover")
                    print(f"[Batch {batch_number}] Skipping batch due to memory constraints")
                    # Continue with next batch rather than crashing
                    continue
            
            finally:
                # Cleanup batch array and intermediate variables
                del batch_array
                if 'masks' in locals():
                    del masks
                if 'coordinates' in locals():
                    del coordinates
                if 'DLC_data_batch' in locals():
                    del DLC_data_batch
                
                # Force GPU cache clearing after batch completion
                torch.cuda.empty_cache()
                gc.collect()
                
                # GPU memory monitoring - batch end
                print_gpu_memory(f"[Batch {batch_number}] End - ")
    else:
        print("Using traditional JPEG extraction processing")
        # Traditional approach: extract frames to disk
        frames_dir, batch_frame_count = create_frames_directory(video_path, batch_size)
        
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

    # After everything is finished, delete the temporary folder (only for traditional method)
    if not use_image_arrays and frames_dir is not None:
        clear_temp_folder(frames_dir)

if __name__ == "__main__":
     main(sys.argv[1:])  # exclude the script name from the args when called from sh
