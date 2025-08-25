"""
Lazy Video Frame Provider for SAM2 Batch Processing

This module provides on-demand video frame loading to maintain 400-frame batch benefits
while avoiding memory overload. Frames are loaded incrementally when SAM2 requests them.
"""

import cv2
import torch
import numpy as np
from typing import Dict, Optional, Tuple, Union
from collections import OrderedDict
from threading import Lock
import gc


class LazyVideoProvider:
    """
    Lazy video frame provider that loads frames on-demand for SAM2.
    
    Maintains LRU cache for recently accessed frames and supports smart prefetching
    to optimize for SAM2's access patterns during temporal propagation.
    """
    
    def __init__(
        self,
        video_path: str,
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
        Initialize the lazy video provider.
        
        Args:
            video_path: Path to the video file
            batch_start_frame: Starting frame index for this batch
            batch_end_frame: Ending frame index for this batch
            image_size: Target image size for SAM2
            device: PyTorch device (CPU or CUDA)
            cache_size: Maximum number of frames to keep in memory
            prefetch_size: Number of frames to prefetch ahead
            img_mean: ImageNet normalization mean
            img_std: ImageNet normalization std
        """
        self.video_path = video_path
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
        self.cache_lock = Lock()
        
        # Video capture object (initialized lazily)
        self._cap: Optional[cv2.VideoCapture] = None
        
        # Video properties (loaded on first access)
        self.video_height: Optional[int] = None
        self.video_width: Optional[int] = None
        
        # Statistics for monitoring
        self.cache_hits = 0
        self.cache_misses = 0
        self.frames_loaded = 0
    
    def _initialize_video_capture(self) -> cv2.VideoCapture:
        """Initialize video capture if not already done."""
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.video_path)
            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to open video: {self.video_path}")
            
            # Get video properties
            self.video_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        return self._cap
    
    def _load_frame_from_video(self, global_frame_idx: int) -> torch.Tensor:
        """
        Load a single frame from the video file.
        
        Args:
            global_frame_idx: Global frame index in the video
            
        Returns:
            Normalized frame tensor of shape (3, image_size, image_size)
        """
        cap = self._initialize_video_capture()
        
        # Seek to the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, global_frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            raise RuntimeError(f"Failed to read frame {global_frame_idx} from {self.video_path}")
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame
        frame_resized = cv2.resize(frame_rgb, (self.image_size, self.image_size))
        
        # Convert to tensor and normalize
        frame_tensor = torch.from_numpy(frame_resized.astype(np.float32) / 255.0).permute(2, 0, 1)
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
                    frame = self._load_frame_from_video(global_idx)
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
        frame = self._load_frame_from_video(global_frame_idx)
        
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
        if self._cap is not None:
            self._cap.release()
        self.clear_cache()


def create_lazy_video_provider(
    video_path: str,
    batch_start_frame: int,
    batch_size: int,
    image_size: int,
    device: torch.device,
    **kwargs
) -> LazyVideoProvider:
    """
    Convenience function to create a LazyVideoProvider.
    
    Args:
        video_path: Path to the video file
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
        video_path=video_path,
        batch_start_frame=batch_start_frame,
        batch_end_frame=batch_end_frame,
        image_size=image_size,
        device=device,
        **kwargs
    )