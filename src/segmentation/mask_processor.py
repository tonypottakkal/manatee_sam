"""
Mask processing utilities for applying waterline and reflection filtering.

This module provides utilities for processing segmentation masks by applying
waterline detection and reflection filtering to produce clean manatee body masks.
"""

import logging
import numpy as np
from typing import Optional, List

from src.models.data_models import SegmentationMask, WaterlineBoundary, ReflectionRegion
from src.detection.waterline_detector import WaterlineDetector
from src.detection.reflection_filter import ReflectionFilter


class MaskProcessor:
    """
    Processes segmentation masks by applying waterline and reflection filtering.
    
    This class integrates waterline detection and reflection filtering to clean up
    segmentation masks, removing underwater portions and reflection artifacts.
    """
    
    def __init__(self, waterline_detector: Optional[WaterlineDetector] = None,
                 reflection_filter: Optional[ReflectionFilter] = None):
        """
        Initialize the mask processor.
        
        Args:
            waterline_detector: Optional waterline detector instance
            reflection_filter: Optional reflection filter instance
        """
        self.waterline_detector = waterline_detector or WaterlineDetector()
        self.reflection_filter = reflection_filter or ReflectionFilter()
    
    def process_mask(self, mask: SegmentationMask, image: np.ndarray) -> SegmentationMask:
        """
        Process a segmentation mask by applying waterline and reflection filtering.
        
        Args:
            mask: Input segmentation mask
            image: Original RGB image
            
        Returns:
            Processed segmentation mask with filtering applied
        """
        if mask is None or image is None:
            return mask
        
        try:
            # Convert mask to binary format for processing
            binary_mask = mask.mask.astype(np.uint8) * 255
            
            # Step 1: Detect waterline
            waterline = self.waterline_detector.detect_waterline(image)
            
            # Step 2: Apply waterline filtering
            if waterline is not None:
                binary_mask = self.waterline_detector.filter_mask_by_waterline(binary_mask, waterline)
                logging.info(f"Applied waterline filtering with confidence {waterline.confidence:.3f}")
                
                # Step 3: Apply reflection filtering
                reflections = self.reflection_filter.detect_reflections(image, waterline)
                if reflections:
                    binary_mask = self.reflection_filter.filter_mask_reflections(binary_mask, reflections)
                    logging.info(f"Applied reflection filtering - removed {len(reflections)} reflections")
            else:
                logging.warning("No waterline detected - skipping waterline and reflection filtering")
            
            # Step 4: Create updated segmentation mask
            processed_mask = SegmentationMask(
                mask=(binary_mask > 0).astype(bool),
                bbox=mask.bbox,  # Keep original bbox for now
                area=int(np.sum(binary_mask > 0)),
                confidence=mask.confidence,  # Keep original confidence
                stability_score=mask.stability_score
            )
            
            return processed_mask
            
        except Exception as e:
            logging.error(f"Error processing mask: {e}")
            return mask  # Return original mask if processing fails
    
    def apply_waterline_filtering(self, mask: np.ndarray, image: np.ndarray) -> tuple[np.ndarray, Optional[WaterlineBoundary]]:
        """
        Apply waterline filtering to a binary mask.
        
        Args:
            mask: Input binary mask
            image: Original RGB image
            
        Returns:
            Tuple of (filtered_mask, waterline_boundary)
        """
        if mask is None or image is None:
            return mask, None
        
        try:
            # Detect waterline
            waterline = self.waterline_detector.detect_waterline(image)
            
            if waterline is None:
                return mask, None
            
            # Apply waterline filtering
            filtered_mask = self.waterline_detector.filter_mask_by_waterline(mask, waterline)
            
            return filtered_mask, waterline
            
        except Exception as e:
            logging.error(f"Error applying waterline filtering: {e}")
            return mask, None
    
    def apply_reflection_filtering(self, mask: np.ndarray, image: np.ndarray, 
                                 waterline: WaterlineBoundary) -> tuple[np.ndarray, List[ReflectionRegion]]:
        """
        Apply reflection filtering to a binary mask.
        
        Args:
            mask: Input binary mask
            image: Original RGB image
            waterline: Detected waterline boundary
            
        Returns:
            Tuple of (filtered_mask, detected_reflections)
        """
        if mask is None or image is None or waterline is None:
            return mask, []
        
        try:
            # Detect reflections
            reflections = self.reflection_filter.detect_reflections(image, waterline)
            
            if not reflections:
                return mask, []
            
            # Apply reflection filtering
            filtered_mask = self.reflection_filter.filter_mask_reflections(mask, reflections)
            
            return filtered_mask, reflections
            
        except Exception as e:
            logging.error(f"Error applying reflection filtering: {e}")
            return mask, []
    
    def create_clean_mask(self, mask: np.ndarray, image: np.ndarray) -> dict:
        """
        Create a clean mask by applying all available filtering techniques.
        
        Args:
            mask: Input binary mask
            image: Original RGB image
            
        Returns:
            Dictionary containing the processed mask and metadata
        """
        result = {
            "mask": mask,
            "waterline": None,
            "reflections": [],
            "processing_applied": [],
            "confidence": 1.0
        }
        
        if mask is None or image is None:
            return result
        
        try:
            current_mask = mask.copy()
            
            # Apply waterline filtering
            filtered_mask, waterline = self.apply_waterline_filtering(current_mask, image)
            if waterline is not None:
                current_mask = filtered_mask
                result["waterline"] = waterline
                result["processing_applied"].append("waterline_filtering")
                result["confidence"] *= waterline.confidence
            
            # Apply reflection filtering if waterline was detected
            if waterline is not None:
                filtered_mask, reflections = self.apply_reflection_filtering(current_mask, image, waterline)
                if reflections:
                    current_mask = filtered_mask
                    result["reflections"] = reflections
                    result["processing_applied"].append("reflection_filtering")
                    # Reduce confidence slightly for each reflection found
                    reflection_penalty = min(0.2, len(reflections) * 0.05)
                    result["confidence"] *= (1.0 - reflection_penalty)
            
            result["mask"] = current_mask
            result["confidence"] = max(0.0, min(1.0, result["confidence"]))
            
            return result
            
        except Exception as e:
            logging.error(f"Error creating clean mask: {e}")
            return result