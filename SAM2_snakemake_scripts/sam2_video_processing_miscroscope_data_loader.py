'''
This script processes TIFF/BTF frames and generates segmentation masks using the SAM2 model with lazy loading.

Main steps:
1. **CUDA Setup**: Checks if CUDA (GPU support) is available and selects the device accordingly.
2. **DeepLabCut (DLC) CSV Extraction**: Reads DLC CSV file containing body part tracking data.
3. **Lazy Frame Loading**: Uses MicroscopeDataReader with dask arrays for on-demand frame loading.
4. **Coordinate Extraction**: Extracts coordinates based on highest likelihood values.
5. **Segmentation with SAM2**: Generates segmentation masks using temporal propagation.
6. **Saving Masks**: Saves binary masks to a TIFF file.
'''

import torch
import gc
import os
import cv2
import numpy as np
import pandas as pd
import sys
import argparse
import tifffile as tiff
from typing import Dict, Optional, Union
from collections import OrderedDict
from threading import Lock, RLock
from imutils import MicroscopeDataReader
import dask.array as da

# Check if CUDA is available and display the GPU information
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    for i in range(gpu_count):
        print(f"CUDA is available. Using GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Using CPU.")


# ============================================================================
# LAZY TIFF/BTF FRAME PROVIDER
# ============================================================================

class LazyVideoProvider:
    """
    Lazy TIFF/BTF frame provider that loads frames on-demand for SAM2.

    Maintains LRU cache for recently accessed frames and supports smart prefetching
    to optimize for SAM2's access patterns during temporal propagation.
    """

    def __init__(
        self,
        tiff_path: str,
        batch_start_frame: int,
        batch_end_frame: int,
        image_size: int,
        device: torch.device,
        cache_size: int = 50,
        prefetch_size: int = 5,
        img_mean: tuple = (0.485, 0.456, 0.406),
        img_std: tuple = (0.229, 0.224, 0.225)
    ):
        """
        Initialize the lazy TIFF/BTF provider.

        Args:
            tiff_path: Path to the TIFF file, BTF file, or folder containing TIFF stack
            batch_start_frame: Starting frame index for this batch
            batch_end_frame: Ending frame index for this batch
            image_size: Target image size for SAM2
            device: PyTorch device (CPU or CUDA)
            cache_size: Maximum number of frames to keep in memory
            prefetch_size: Number of frames to prefetch ahead
            img_mean: ImageNet normalization mean
            img_std: ImageNet normalization std
        """
        self.tiff_path = tiff_path
        self.batch_start_frame = batch_start_frame
        self.batch_end_frame = batch_end_frame
        self.batch_size = batch_end_frame - batch_start_frame
        self.image_size = image_size
        self.device = device
        self.cache_size = cache_size
        self.prefetch_size = prefetch_size

        # Normalization parameters
        self.img_mean = torch.tensor(img_mean, dtype=torch.float32, device=device)[:, None, None]
        self.img_std = torch.tensor(img_std, dtype=torch.float32, device=device)[:, None, None]

        # LRU cache for frames (OrderedDict maintains access order)
        self.frame_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.cache_lock = RLock()  # Use RLock (reentrant) to avoid deadlock in prefetch

        # MicroscopeDataReader object (initialized lazily)
        self._reader: Optional[MicroscopeDataReader] = None
        self._dask_array: Optional[da.Array] = None

        # Image properties (loaded on first access)
        self.image_height: Optional[int] = None
        self.image_width: Optional[int] = None

        # SAM2 compatibility - these properties are expected by SAM2
        self.video_height: Optional[int] = None
        self.video_width: Optional[int] = None

        # Statistics for monitoring
        self.cache_hits = 0
        self.cache_misses = 0
        self.frames_loaded = 0

    def _initialize_microscope_reader(self):
        """Initialize MicroscopeDataReader if not already done."""
        if self._reader is None:
            # Detect if path is a folder or BTF file
            if os.path.isdir(self.tiff_path):
                # Folder path - use TIFF logic
                print(f"Loading TIFF stack from folder: {self.tiff_path}")
                self._reader = MicroscopeDataReader(self.tiff_path, as_raw_tiff=False)
            else:
                # BTF file path
                print(f"Loading BTF file: {self.tiff_path}")
                try:
                    self._reader = MicroscopeDataReader(self.tiff_path, as_raw_tiff=True, raw_tiff_num_slices=1)
                except TypeError:
                    # Fallback to non-raw TIFF
                    self._reader = MicroscopeDataReader(self.tiff_path, as_raw_tiff=False)

            # Get the dask array
            self._dask_array = da.squeeze(self._reader.dask_array)

            # Get image properties from first frame
            if len(self._dask_array.shape) == 3:
                # 3D array: (frames, height, width)
                _, self.image_height, self.image_width = self._dask_array.shape
            elif len(self._dask_array.shape) == 2:
                # 2D array: (height, width) - single frame
                self.image_height, self.image_width = self._dask_array.shape
            else:
                raise ValueError(f"Unexpected dask array shape: {self._dask_array.shape}")

            # Set SAM2 compatibility properties
            self.video_height = self.image_height
            self.video_width = self.image_width

            print(f"Loaded TIFF/BTF with shape: {self._dask_array.shape}, dtype: {self._dask_array.dtype}")

    def _load_frame_from_tiff(self, global_frame_idx: int) -> torch.Tensor:
        """
        Load a single frame from the TIFF/BTF file via dask array.

        Args:
            global_frame_idx: Global frame index in the stack

        Returns:
            Normalized frame tensor of shape (3, image_size, image_size)
        """
        if self._dask_array is None:
            self._initialize_microscope_reader()

        # Load frame from dask array (this is where the lazy loading happens!)
        frame = np.array(self._dask_array[global_frame_idx])

        # Ensure frame is 2D
        if frame.ndim != 2:
            raise ValueError(f"Expected 2D frame, got shape {frame.shape}")

        # Convert grayscale to RGB by repeating channels
        frame_rgb = np.stack([frame, frame, frame], axis=-1)

        # Normalize to 0-1 range if needed
        if frame_rgb.dtype == np.uint8:
            frame_rgb = frame_rgb.astype(np.float32) / 255.0
        elif frame_rgb.dtype == np.uint16:
            frame_rgb = frame_rgb.astype(np.float32) / 65535.0
        else:
            # Assume already in float format
            frame_rgb = frame_rgb.astype(np.float32)
            # Normalize if values are outside [0, 1]
            if frame_rgb.max() > 1.0:
                frame_rgb = frame_rgb / frame_rgb.max()

        # Resize frame
        frame_resized = cv2.resize(frame_rgb, (self.image_size, self.image_size))

        # Convert to tensor and normalize
        frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1)
        frame_tensor = frame_tensor.to(self.device)

        # Apply ImageNet normalization
        frame_tensor = (frame_tensor - self.img_mean) / self.img_std

        self.frames_loaded += 1
        return frame_tensor

    def _manage_cache_size(self):
        """Remove oldest frames from cache if it exceeds cache_size."""
        with self.cache_lock:
            while len(self.frame_cache) > self.cache_size:
                # Remove least recently used frame
                oldest_idx, oldest_frame = self.frame_cache.popitem(last=False)
                # Explicit deletion to help with memory management
                del oldest_frame

    def _prefetch_frames(self, current_idx: int):
        """
        Prefetch upcoming frames for better performance.

        Args:
            current_idx: Current batch-relative frame index
        """
        # Prefetch next few frames if they're within the batch
        for offset in range(1, self.prefetch_size + 1):
            next_idx = current_idx + offset
            if next_idx < self.batch_size and next_idx not in self.frame_cache:
                try:
                    global_idx = self.batch_start_frame + next_idx
                    frame = self._load_frame_from_tiff(global_idx)
                    with self.cache_lock:
                        self.frame_cache[next_idx] = frame
                        # Move to end (most recently used)
                        self.frame_cache.move_to_end(next_idx)
                    # Manage cache size after adding
                    self._manage_cache_size()
                except Exception as e:
                    # Don't fail prefetching, just log and continue
                    print(f"Prefetch failed for frame {next_idx}: {e}")
                    break

    def __getitem__(self, batch_relative_idx: int) -> torch.Tensor:
        """
        Get a frame by batch-relative index (SAM2 interface).

        Args:
            batch_relative_idx: Frame index relative to batch start (0 to batch_size-1)

        Returns:
            Normalized frame tensor of shape (3, image_size, image_size)
        """
        if not (0 <= batch_relative_idx < self.batch_size):
            raise IndexError(f"Frame index {batch_relative_idx} out of range [0, {self.batch_size})")

        # Check cache first
        with self.cache_lock:
            if batch_relative_idx in self.frame_cache:
                # Move to end (mark as recently used)
                self.frame_cache.move_to_end(batch_relative_idx)
                self.cache_hits += 1
                frame = self.frame_cache[batch_relative_idx]

                # Prefetch in background (don't block current request)
                self._prefetch_frames(batch_relative_idx)
                return frame

        # Cache miss - load frame
        self.cache_misses += 1
        global_frame_idx = self.batch_start_frame + batch_relative_idx
        frame = self._load_frame_from_tiff(global_frame_idx)

        # Add to cache
        with self.cache_lock:
            self.frame_cache[batch_relative_idx] = frame
            # Move to end (most recently used)
            self.frame_cache.move_to_end(batch_relative_idx)

        # Manage cache size and prefetch
        self._manage_cache_size()
        self._prefetch_frames(batch_relative_idx)

        return frame

    def __len__(self) -> int:
        """Return the number of frames in this batch."""
        return self.batch_size

    def to(self, device: torch.device) -> 'LazyVideoProvider':
        """
        Move the provider to a different device.
        This mainly affects where new frames are loaded.
        """
        if device != self.device:
            self.device = device
            self.img_mean = self.img_mean.to(device)
            self.img_std = self.img_std.to(device)

            # Move cached frames to new device
            with self.cache_lock:
                for idx in list(self.frame_cache.keys()):
                    self.frame_cache[idx] = self.frame_cache[idx].to(device)

        return self

    def clear_cache(self):
        """Clear the frame cache to free memory."""
        with self.cache_lock:
            # Explicit deletion of all cached frames
            for frame in self.frame_cache.values():
                del frame
            self.frame_cache.clear()

        # Force garbage collection
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'frames_loaded': self.frames_loaded,
            'cached_frames': len(self.frame_cache),
            'cache_size_limit': self.cache_size
        }

    def __del__(self):
        """Cleanup when the provider is destroyed."""
        self.clear_cache()


def create_lazy_video_provider(
    tiff_path: str,
    batch_start_frame: int,
    batch_size: int,
    image_size: int,
    device: torch.device,
    **kwargs
) -> LazyVideoProvider:
    """
    Convenience function to create a LazyVideoProvider for TIFF/BTF files.

    Args:
        tiff_path: Path to the TIFF file, BTF file, or folder
        batch_start_frame: Starting frame index for this batch
        batch_size: Number of frames in the batch
        image_size: Target image size for SAM2
        device: PyTorch device
        **kwargs: Additional arguments for LazyVideoProvider

    Returns:
        Configured LazyVideoProvider instance
    """
    batch_end_frame = batch_start_frame + batch_size

    return LazyVideoProvider(
        tiff_path=tiff_path,
        batch_start_frame=batch_start_frame,
        batch_end_frame=batch_end_frame,
        image_size=image_size,
        device=device,
        **kwargs
    )


def init_state_with_lazy_provider(
    predictor,
    video_provider,
    offload_video_to_cpu=False,
    offload_state_to_cpu=False,
):
    """
    Initialize an inference state using a LazyVideoProvider.

    This is a 1:1 copy of SAM2VideoPredictor.init_state() but uses lazy loading
    instead of loading all frames at once with load_video_frames().

    Args:
        predictor: SAM2VideoPredictor instance
        video_provider: LazyVideoProvider instance (already initialized)
        offload_video_to_cpu: Whether to offload video frames to CPU
        offload_state_to_cpu: Whether to offload inference state to CPU

    Returns:
        inference_state dict ready for SAM2 processing
    """
    compute_device = predictor.device  # device of the model

    # Make sure video_provider has loaded its properties
    if video_provider.video_height is None:
        video_provider._initialize_microscope_reader()

    # Instead of calling load_video_frames(), use the lazy provider directly
    images = video_provider
    video_height = video_provider.video_height
    video_width = video_provider.video_width

    # Everything below is a 1:1 copy from SAM2VideoPredictor.init_state()
    inference_state = {}
    inference_state["images"] = images
    inference_state["num_frames"] = len(images)
    # whether to offload the video frames to CPU memory
    # turning on this option saves the GPU memory with only a very small overhead
    inference_state["offload_video_to_cpu"] = offload_video_to_cpu
    # whether to offload the inference state to CPU memory
    # turning on this option saves the GPU memory at the cost of a lower tracking fps
    # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
    # and from 24 to 21 when tracking two objects)
    inference_state["offload_state_to_cpu"] = offload_state_to_cpu
    # the original video height and width, used for resizing final output scores
    inference_state["video_height"] = video_height
    inference_state["video_width"] = video_width
    inference_state["device"] = compute_device
    if offload_state_to_cpu:
        inference_state["storage_device"] = torch.device("cpu")
    else:
        inference_state["storage_device"] = compute_device
    # inputs on each frame
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    # visual features on a small number of recently visited frames for quick interactions
    inference_state["cached_features"] = {}
    # values that don't change across frames (so we only need to hold one copy of them)
    inference_state["constants"] = {}
    # mapping between client-side object id and model-side object index
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    # A storage to hold the model's tracking results and states on each frame
    inference_state["output_dict"] = {
        "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
    }
    # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
    inference_state["output_dict_per_obj"] = {}
    # A temporary storage to hold new outputs when user interact with a frame
    # to add clicks or mask (it's merged into "output_dict" before propagation starts)
    inference_state["temp_output_dict_per_obj"] = {}
    # Frames that already holds consolidated outputs from click or mask inputs
    # (we directly use their consolidated outputs during tracking)
    inference_state["consolidated_frame_inds"] = {
        "cond_frame_outputs": set(),  # set containing frame indices
        "non_cond_frame_outputs": set(),  # set containing frame indices
    }
    # metadata for each tracking frame (e.g. which direction it's tracked)
    inference_state["tracking_has_started"] = False
    inference_state["frames_already_tracked"] = {}
    # Warm up the visual backbone and cache the image feature on frame 0
    predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)
    return inference_state


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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
    """Read and format DeepLabCut CSV file."""
    df = pd.read_csv(csv_file_path)

    # Remove column names and set first row to new column name
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
    """Extract coordinates based on highest average likelihood."""
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

    # Extract x and y coordinates as list
    result = {}
    for bodypart in bodyparts:
        if bodypart in selected_row.columns.get_level_values(0):
            x_values = pd.to_numeric(selected_row[bodypart]['x'], errors='coerce')
            y_values = pd.to_numeric(selected_row[bodypart]['y'], errors='coerce')
            result[bodypart] = list(zip(x_values, y_values))

    return result, selected_index


def process_mask(mask):
    """Process mask to ensure it's in correct format (binary, 8-bit)."""
    if mask.dtype == bool:
        binary_mask = (mask * 255).astype(np.uint8)
    elif mask.dtype == np.uint8:
        binary_mask = mask
    else:
        binary_mask = ((mask > 0) * 255).astype(np.uint8)
    return binary_mask


# ============================================================================
# SAM2 SEGMENTATION FUNCTION
# ============================================================================

def segment_object_lazy(predictor, video_provider, coordinates, frame_number, batch_start, batch_size):
    """
    Generate segmentation masks using SAM2 with lazy video provider.

    Args:
        predictor: SAM2 predictor instance
        video_provider: LazyVideoProvider instance
        coordinates: Dictionary of body part coordinates
        frame_number: Frame index for initial annotation (global frame index)
        batch_start: Starting frame index for this batch (global frame index)
        batch_size: Number of frames in batch

    Returns:
        Dictionary mapping frame indices to binary masks
    """
    # Ensure video provider has the required properties initialized
    if video_provider.video_height is None:
        video_provider._initialize_microscope_reader()

    # Use our custom init function instead of monkey-patching
    inference_state = init_state_with_lazy_provider(
        predictor=predictor,
        video_provider=video_provider,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False
    )
    print("Initialized inference state with lazy video provider")

    # Convert global frame number to batch-relative index
    batch_relative_frame_number = frame_number - batch_start

    # Prepare points and labels like the working script
    points_list = []
    labels_list = []

    for bodypart, coords in coordinates.items():
        for x, y in coords:
            print(f"Adding point for {bodypart}: ({x:.2f}, {y:.2f})")
            points_list.append([x, y])
            labels_list.append(1)

    # Convert lists to numpy arrays
    points = np.array(points_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)

    print("Points array:", points)
    print("Labels array:", labels)

    # Add points to model using batch-relative frame number
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=batch_relative_frame_number,  # Use batch-relative index
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

                # Add the binary mask to video_segments
                video_segments[out_frame_idx] = binary_mask
                print(f"Processed frame {out_frame_idx}")
            else:
                # Create a black mask (all zeros) with the same width and height as the original mask
                black_mask = np.zeros(mask.shape[1:], dtype=np.uint8)

                # Add the black mask to video_segments
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

                # Add the binary mask to video_segments
                video_segments[out_frame_idx] = binary_mask
                print(f"Processed frame {out_frame_idx}")
            else:
                # Create a black mask (all zeros) with the same width and height as the original mask
                black_mask = np.zeros(mask.shape[1:], dtype=np.uint8)

                # Add the black mask to video_segments
                video_segments[out_frame_idx] = black_mask
                print(f"Non-2D mask for frame {out_frame_idx}. Added black mask instead.")

    # Sort the dictionary to ensure the frames are in proper order
    video_segments = dict(sorted(video_segments.items()))

    # Clean up inference state
    predictor.reset_state(inference_state)

    return video_segments


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def main(args=None):
    """Main processing function — lazy read + streaming write to single BigTIFF (.btf)."""
    import sys
    import gc
    import os
    from pathlib import Path
    import argparse
    import torch
    import tifffile as tiff

    # --- parse CLI ---
    parser = argparse.ArgumentParser(description='SAM2 TIFF/BTF Lazy Video Processing Pipeline (stream to BigTIFF)')
    parser.add_argument('-tiff_path', type=str, required=True,
                        help='Path to the input TIFF/BTF file or folder')
    parser.add_argument('-output_file_path', type=str, required=True,
                        help='Output .btf (BigTIFF) path')
    parser.add_argument('-DLC_csv_file_path', type=str, required=True,
                        help='Path to the DeepLabCut CSV file')
    parser.add_argument('-column_names', type=str, required=True,
                        help='Comma-separated list of body part column names')
    parser.add_argument('-SAM2_path', type=str, required=True,
                        help='Path to the SAM2 model directory')
    parser.add_argument('--batch_size', type=int, default=400,
                        help='Number of frames per batch')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device ID')
    args = parser.parse_args(args)

    # --- debug / echo inputs ---
    print(f"\n{'='*60}")
    print(f"DEBUG: Input path: {args.tiff_path}")
    print(f"DEBUG: DLC CSV    : {args.DLC_csv_file_path}")
    print(f"DEBUG: Output BTF : {args.output_file_path}")
    print(f"DEBUG: Using MicroscopeDataReader for lazy loading")
    print(f"{'='*60}\n")

    # --- quick input validation ---
    in_path = Path(args.tiff_path)
    if not (in_path.is_file() or in_path.is_dir()):
        raise FileNotFoundError(f"-tiff_path not found: {args.tiff_path}")
    if in_path.is_file() and in_path.suffix.lower() not in {'.tif', '.tiff', '.btf'}:
        raise ValueError(f"-tiff_path expects TIFF/BTF (got {in_path.suffix}).")

    out_path = Path(args.output_file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- parse columns ---
    column_names = [name.strip() for name in args.column_names.split(',')]

    # --- device selection ---
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"CUDA available. GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA not available. Using CPU.")
    print(f"Using device: {device}")

    # --- SAM2 init ---
    print("Initializing SAM2 model...")
    sys.path.append(args.SAM2_path)
    from sam2.build_sam import build_sam2_video_predictor
    sam2_checkpoint = os.path.join(args.SAM2_path, "checkpoints", "sam2_hiera_large.pt")
    model_cfg = "sam2_hiera_l.yaml"  # Just filename - Hydra knows to look in sam2_configs
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    print("SAM2 model loaded successfully")

    # --- DLC load ---
    print(f"Reading DLC CSV: {args.DLC_csv_file_path}")
    DLC_data = read_DLC_csv(args.DLC_csv_file_path)
    total_frames = len(DLC_data)
    print(f"Total frames in DLC data: {total_frames}")

    # --- batches ---
    num_batches = (total_frames + args.batch_size - 1) // args.batch_size
    print(f"Processing {num_batches} batches with batch size {args.batch_size}")

    # --- streaming BigTIFF writer: open once, append pages as we go ---
    print("\n" + "="*60)
    print("Saving masks to single BigTIFF (.btf) — streaming one frame at a time...")
    print("="*60)

    # overwrite existing output if present
    if out_path.exists():
        out_path.unlink()

    with tiff.TiffWriter(out_path.as_posix(), bigtiff=True) as tif_writer:

        print("Using lazy TIFF/BTF loading")

        for batch_number in range(num_batches):
            batch_start = batch_number * args.batch_size
            batch_end = min(batch_start + args.batch_size, total_frames)
            current_batch_size = batch_end - batch_start

            print(f"\n{'='*60}")
            print(f"Processing Batch {batch_number + 1}/{num_batches}")
            print(f"Frames: {batch_start} to {batch_end - 1} (size: {current_batch_size})")
            print(f"{'='*60}")

            print_gpu_memory(f"[Lazy Batch {batch_number}] Start - ")

            # lazy provider for this batch
            video_provider = create_lazy_video_provider(
                tiff_path=args.tiff_path,
                batch_start_frame=batch_start,
                batch_size=current_batch_size,
                image_size=1024,  # SAM2 standard
                device=device,
            )
            if video_provider.video_height is None:
                video_provider._initialize_microscope_reader()

            # slice DLC for batch & pick coordinates
            DLC_data_batch = DLC_data.iloc[batch_start:batch_end]
            coordinates, frame_number = extract_coordinate_by_likelihood(DLC_data_batch, column_names)
            print(f"Selected frame {frame_number} for annotation in batch {batch_number}")

            # segment + propagate
            print(f"Generating masks for batch {batch_number}...")
            masks = segment_object_lazy(
                predictor=predictor,
                video_provider=video_provider,
                coordinates=coordinates,
                frame_number=frame_number,
                batch_start=batch_start,
                batch_size=current_batch_size,
            )
            print(f"Masks generated for batch {batch_number}. Number of masks: {len(masks)}")

            # --- stream masks directly into the BigTIFF ---
            # NOTE: 'contiguous=True' is okay since all masks share shape & dtype.
            #       You can also add compression='zlib' if you like smaller files.
            for out_frame_idx in sorted(masks.keys()):
                global_frame_idx = out_frame_idx + batch_start
                processed_mask = process_mask(masks[out_frame_idx])  # 2D uint8
                tif_writer.write(
                    processed_mask,
                    contiguous=True,
                    # compression='zlib',  # optional: enable if you want compression
                    description=f"GlobalFrameIndex={int(global_frame_idx)};Batch={int(batch_number)}",
                )

            # cache stats
            stats = video_provider.get_stats()
            print(f"Cache stats - Hits: {stats['cache_hits']}, Misses: {stats['cache_misses']}, Hit rate: {stats['hit_rate']:.2%}")

            # per-batch cleanup
            video_provider.clear_cache()
            del video_provider, masks, coordinates, DLC_data_batch
            torch.cuda.empty_cache()
            gc.collect()
            print_gpu_memory(f"[Lazy Batch {batch_number}] End - ")

    print(f"\nAll masks saved to BigTIFF: {out_path}")
    print("Processing complete!")


if __name__ == "__main__":
    main(sys.argv[1:])


# ============================================================================
# SNAKEMAKE RULE EXAMPLE
# ============================================================================
"""
rule sam2_segment:
    input:
        "{dataset}/{track_dir}/output/.meta.json",
        tiff     = "{dataset}/{track_dir}/output/track.tif",
        dlc_csv = "{dataset}/{track_dir}/output/track" + config["network_string"] + "_filtered.csv"
    output:
        mask = "{dataset}/{track_dir}/output/track_mask.btf"
    threads: create_static_threads_function("sam2_segment")
    resources:
        mem_mb=create_static_memory_function("sam2_segment"),
        time=create_static_time_function("sam2_segment"),
        partition=create_partition_function("sam2_segment"),
        gres=create_gres_function("sam2_segment")
    params:
        column_names      = ["neck"],
        model_path        = "/lisc/scratch/neurobiology/zimmer/schaar/code/github/segment-anything-2",
        sam2_conda_env_name = "/lisc/scratch/neurobiology/zimmer/.conda/envs/SAM2_shared",
        batch_size        = config["batch_size"]
    shell:
        '''
        # I started getting an error with the xml_catalog_files_libxml2 variable, so check if it is set
        if [ -z "${{xml_catalog_files_libxml2:-}}" ]; then
            export xml_catalog_files_libxml2=""
        fi

        module load cuda-toolkit/12.6.3
        
        # Activate the environment and the correct cuda
        source /lisc/app/conda/miniforge3/bin/activate {params.sam2_conda_env_name}

        # Run the script directly without temp directory overhead
        python -c "from SAM2_snakemake_scripts.sam2_tiff_lazy_processing import main; main(['-tiff_path', '{input.tiff}', '-output_file_path', '{output.mask}', '-DLC_csv_file_path', '{input.dlc_csv}', '-column_names', '{params.column_names}', '-SAM2_path', '{params.model_path}', '--batch_size', '{params.batch_size}', '--device', '${{CUDA_VISIBLE_DEVICES:-0}}'])"
        '''
"""