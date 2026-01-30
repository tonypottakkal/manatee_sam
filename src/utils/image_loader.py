"""
Image loading and validation utilities for the manatee segmentation system.
"""

import os
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image

from src.models.data_models import ImageInfo


class ImageLoader:
    """
    Handles PNG image loading and validation with error handling.
    
    This class provides functionality to load PNG images, validate their format,
    and extract metadata while preserving original dimensions and color information.
    """
    
    def __init__(self):
        """Initialize the ImageLoader."""
        pass
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load a PNG image from the specified path.
        
        Args:
            image_path: Path to the PNG image file
            
        Returns:
            np.ndarray: Loaded image as numpy array
            
        Raises:
            FileNotFoundError: If the image path does not exist
            ValueError: If the file is not a valid PNG image
            IOError: If there's an error reading the file
        """
        if not self.validate_format(image_path):
            raise ValueError(f"Invalid image format or path: {image_path}")
        
        try:
            # Load image using PIL and convert to numpy array
            with Image.open(image_path) as img:
                # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
                if img.mode == 'RGBA':
                    # Keep RGBA for transparency support
                    image_array = np.array(img)
                elif img.mode == 'L':
                    # Keep grayscale as single channel
                    image_array = np.array(img)
                else:
                    # Convert other modes to RGB
                    img_rgb = img.convert('RGB')
                    image_array = np.array(img_rgb)
                
                return image_array
                
        except Exception as e:
            raise IOError(f"Error loading image {image_path}: {str(e)}")
    
    def validate_format(self, image_path: str) -> bool:
        """
        Validate that the file exists and is a PNG image.
        
        Args:
            image_path: Path to validate
            
        Returns:
            bool: True if valid PNG file, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                return False
            
            # Check file extension
            path_obj = Path(image_path)
            if path_obj.suffix.lower() != '.png':
                return False
            
            # Try to open with PIL to validate it's a valid image
            with Image.open(image_path) as img:
                # Verify it's actually a PNG
                if img.format != 'PNG':
                    return False
                
                # Basic validation - ensure it has valid dimensions
                if img.width <= 0 or img.height <= 0:
                    return False
                    
            return True
            
        except Exception:
            return False
    
    def get_image_info(self, image: np.ndarray) -> ImageInfo:
        """
        Extract metadata information from a loaded image.
        
        Args:
            image: Loaded image as numpy array
            
        Returns:
            ImageInfo: Metadata about the image
        """
        height, width = image.shape[:2]
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        dtype = str(image.dtype)
        file_size = image.nbytes
        
        return ImageInfo(
            width=width,
            height=height,
            channels=channels,
            dtype=dtype,
            file_size=file_size
        )
    
    def get_image_info_from_path(self, image_path: str) -> ImageInfo:
        """
        Get image information directly from file path without loading full image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            ImageInfo: Metadata about the image
            
        Raises:
            FileNotFoundError: If the image path does not exist
            ValueError: If the file is not a valid PNG image
        """
        if not self.validate_format(image_path):
            raise ValueError(f"Invalid image format or path: {image_path}")
        
        try:
            with Image.open(image_path) as img:
                # Get file size
                file_size = os.path.getsize(image_path)
                
                # Determine channels based on mode
                if img.mode == 'L':
                    channels = 1
                elif img.mode == 'RGB':
                    channels = 3
                elif img.mode == 'RGBA':
                    channels = 4
                else:
                    # Convert to RGB to determine channels
                    channels = 3
                
                # Determine dtype (PIL typically uses uint8)
                dtype = 'uint8'
                
                return ImageInfo(
                    width=img.width,
                    height=img.height,
                    channels=channels,
                    dtype=dtype,
                    file_size=file_size
                )
                
        except Exception as e:
            raise IOError(f"Error reading image info from {image_path}: {str(e)}")