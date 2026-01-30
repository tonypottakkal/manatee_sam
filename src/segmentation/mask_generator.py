"""
Mask generation and output functionality for the manatee segmentation system.

This module provides the MaskGenerator class that creates binary masks from segmentation
results and handles PNG output with proper error handling and dimension preservation.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image

from src.models.data_models import SegmentationMask, WaterlineBoundary, ProcessingResult
from src.segmentation.quality_validator import QualityValidator


class MaskGenerator:
    """
    Generates binary masks and handles output formatting for manatee segmentation.
    
    This class creates binary masks where manatee pixels are white (255) and background
    pixels are black (0), with proper PNG output functionality and dimension preservation.
    """
    
    def __init__(self, quality_validator: Optional[QualityValidator] = None):
        """Initialize the mask generator."""
        self.logger = logging.getLogger(__name__)
        self.quality_validator = quality_validator or QualityValidator()
    
    def create_binary_mask(self, segmentation: SegmentationMask, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create a binary mask from segmentation results.
        
        Args:
            segmentation: Segmentation mask with manatee regions
            image_shape: Original image shape (height, width)
            
        Returns:
            Binary mask array where manatee pixels are 255 and background is 0
            
        Raises:
            ValueError: If segmentation or image_shape is invalid
        """
        if segmentation is None:
            raise ValueError("Segmentation cannot be None")
        
        if not segmentation.validate():
            raise ValueError("Invalid segmentation mask")
        
        if len(image_shape) != 2 or image_shape[0] <= 0 or image_shape[1] <= 0:
            raise ValueError("Invalid image shape")
        
        try:
            height, width = image_shape
            
            # Create empty binary mask
            binary_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Ensure segmentation mask matches image dimensions
            seg_mask = segmentation.mask
            if seg_mask.shape != (height, width):
                self.logger.warning(f"Segmentation mask shape {seg_mask.shape} doesn't match image shape {image_shape}")
                # Resize segmentation mask to match image dimensions
                seg_mask_pil = Image.fromarray((seg_mask * 255).astype(np.uint8))
                seg_mask_pil = seg_mask_pil.resize((width, height), Image.NEAREST)
                seg_mask = np.array(seg_mask_pil) > 0
            
            # Set manatee pixels to white (255)
            binary_mask[seg_mask] = 255
            
            self.logger.info(f"Created binary mask with {np.sum(seg_mask)} manatee pixels")
            return binary_mask
            
        except Exception as e:
            self.logger.error(f"Error creating binary mask: {e}")
            raise ValueError(f"Failed to create binary mask: {e}")
    
    def apply_waterline_filter(self, mask: np.ndarray, waterline: WaterlineBoundary) -> np.ndarray:
        """
        Apply waterline filtering to exclude underwater portions from the mask.
        
        Args:
            mask: Input binary mask
            waterline: Detected waterline boundary
            
        Returns:
            Filtered mask with underwater portions removed
            
        Raises:
            ValueError: If mask or waterline is invalid
        """
        if mask is None:
            raise ValueError("Mask cannot be None")
        
        if waterline is None:
            self.logger.warning("No waterline provided - returning original mask")
            return mask.copy()
        
        if not waterline.validate():
            raise ValueError("Invalid waterline boundary")
        
        try:
            filtered_mask = mask.copy()
            height, width = mask.shape
            
            # Extract line equation coefficients
            a, b, c = waterline.equation
            
            # Create coordinate grids
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            
            # Calculate line values for all pixels
            # Line equation: ax + by + c = 0
            # Points below line have ax + by + c > 0 (assuming water_region_below=True)
            line_values = a * x_coords + b * y_coords + c
            
            # Determine which side is water based on water_region_below flag
            if waterline.water_region_below:
                # Water is below the line (positive line values)
                underwater_mask = line_values > 0
            else:
                # Water is above the line (negative line values)
                underwater_mask = line_values < 0
            
            # Remove underwater portions
            filtered_mask[underwater_mask] = 0
            
            removed_pixels = np.sum(mask) - np.sum(filtered_mask)
            self.logger.info(f"Waterline filtering removed {removed_pixels} underwater pixels")
            
            return filtered_mask
            
        except Exception as e:
            self.logger.error(f"Error applying waterline filter: {e}")
            raise ValueError(f"Failed to apply waterline filter: {e}")
    
    def save_mask(self, mask: np.ndarray, output_path: str) -> bool:
        """
        Save binary mask as PNG file with proper error handling.
        
        Args:
            mask: Binary mask array to save
            output_path: Path where to save the PNG file
            
        Returns:
            True if save was successful, False otherwise
            
        Raises:
            ValueError: If mask or output_path is invalid
        """
        if mask is None:
            raise ValueError("Mask cannot be None")
        
        if not output_path or not isinstance(output_path, str):
            raise ValueError("Invalid output path")
        
        try:
            # Ensure mask is in correct format (0-255 uint8)
            if mask.dtype != np.uint8:
                if mask.dtype == bool:
                    save_mask = (mask * 255).astype(np.uint8)
                else:
                    # Normalize to 0-255 range
                    save_mask = ((mask - mask.min()) / (mask.max() - mask.min()) * 255).astype(np.uint8)
            else:
                save_mask = mask.copy()
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                self.logger.info(f"Created output directory: {output_dir}")
            
            # Save as PNG using PIL
            mask_image = Image.fromarray(save_mask, mode='L')  # 'L' for grayscale
            mask_image.save(output_path, 'PNG')
            
            # Verify file was created and has reasonable size
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                self.logger.info(f"Successfully saved mask to {output_path} ({file_size} bytes)")
                return True
            else:
                self.logger.error(f"File was not created at {output_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving mask to {output_path}: {e}")
            return False
    
    def generate_mask_with_metadata(self, segmentation: SegmentationMask, 
                                  image_shape: Tuple[int, int],
                                  waterline: Optional[WaterlineBoundary] = None,
                                  output_path: Optional[str] = None,
                                  original_image: Optional[np.ndarray] = None) -> ProcessingResult:
        """
        Generate a complete binary mask with optional waterline filtering and save to file.
        
        Args:
            segmentation: Segmentation mask with manatee regions
            image_shape: Original image shape (height, width)
            waterline: Optional waterline boundary for filtering
            output_path: Optional path to save the mask
            original_image: Optional original image for quality validation
            
        Returns:
            ProcessingResult with success status, mask, and metadata
        """
        import time
        start_time = time.time()
        
        try:
            # Create binary mask
            binary_mask = self.create_binary_mask(segmentation, image_shape)
            
            # Apply waterline filtering if provided
            if waterline is not None:
                binary_mask = self.apply_waterline_filter(binary_mask, waterline)
            
            # Save to file if path provided
            save_success = True
            if output_path is not None:
                save_success = self.save_mask(binary_mask, output_path)
                if not save_success:
                    return ProcessingResult(
                        success=False,
                        mask=binary_mask,
                        confidence=segmentation.confidence,
                        processing_time=time.time() - start_time,
                        error_message=f"Failed to save mask to {output_path}"
                    )
            
            # Calculate final confidence with quality validation
            coherence_score = self.quality_validator.validate_coherence(binary_mask)
            final_confidence = self.quality_validator.calculate_confidence_score(
                segmentation, binary_mask, original_image
            )
            
            # Check if manual review is needed
            quality_flags = self.quality_validator.flag_for_manual_review(
                final_confidence, coherence_score, binary_mask
            )
            
            processing_time = time.time() - start_time
            
            # Create result with quality information
            result = ProcessingResult(
                success=True,
                mask=binary_mask,
                confidence=final_confidence,
                processing_time=processing_time,
                error_message=None
            )
            
            # Add quality information to result (extend ProcessingResult if needed)
            if hasattr(result, 'quality_flags'):
                result.quality_flags = quality_flags
            
            # Log quality assessment
            if quality_flags["needs_review"]:
                self.logger.warning(f"Segmentation flagged for manual review: {quality_flags['reasons']}")
            else:
                self.logger.info(f"Segmentation passed quality validation (confidence: {final_confidence:.3f})")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error generating mask: {e}")
            
            return ProcessingResult(
                success=False,
                mask=None,
                confidence=0.0,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def validate_output_dimensions(self, mask: np.ndarray, expected_shape: Tuple[int, int]) -> bool:
        """
        Validate that output mask has the expected dimensions.
        
        Args:
            mask: Generated mask to validate
            expected_shape: Expected (height, width) dimensions
            
        Returns:
            True if dimensions match, False otherwise
        """
        if mask is None:
            return False
        
        if len(expected_shape) != 2:
            return False
        
        return mask.shape == expected_shape
    
    def get_mask_statistics(self, mask: np.ndarray) -> dict:
        """
        Get statistics about the generated mask.
        
        Args:
            mask: Binary mask to analyze
            
        Returns:
            Dictionary with mask statistics
        """
        if mask is None:
            return {"error": "Mask is None"}
        
        try:
            stats = {
                "shape": mask.shape,
                "total_pixels": mask.size,
                "manatee_pixels": int(np.sum(mask > 0)),
                "background_pixels": int(np.sum(mask == 0)),
                "manatee_percentage": float(np.sum(mask > 0) / mask.size * 100),
                "dtype": str(mask.dtype),
                "min_value": int(mask.min()),
                "max_value": int(mask.max())
            }
            
            return stats
            
        except Exception as e:
            return {"error": f"Failed to calculate statistics: {e}"}
    
    def generate_quality_report(self, segmentation: SegmentationMask,
                              final_mask: np.ndarray,
                              original_image: Optional[np.ndarray] = None) -> dict:
        """
        Generate a comprehensive quality report for the segmentation.
        
        Args:
            segmentation: Original segmentation mask with metadata
            final_mask: Final processed binary mask
            original_image: Optional original image for enhanced validation
            
        Returns:
            Dictionary with comprehensive quality metrics and recommendations
        """
        try:
            return self.quality_validator.generate_quality_report(
                segmentation, final_mask, original_image
            )
        except Exception as e:
            self.logger.error(f"Error generating quality report: {e}")
            return {
                "overall_quality": "error",
                "error": str(e),
                "confidence_score": 0.0,
                "coherence_score": 0.0
            }