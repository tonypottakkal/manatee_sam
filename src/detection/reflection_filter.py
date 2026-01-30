"""
Reflection filtering module for manatee segmentation system.

This module implements reflection detection and filtering using symmetric pattern detection,
brightness and color similarity analysis, and morphological operations for boundary cleanup.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from scipy import ndimage
from skimage.feature import match_template
from skimage.morphology import binary_opening, binary_closing, disk

# Use absolute imports to avoid relative import issues
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_models import ReflectionRegion, WaterlineBoundary


class ReflectionFilter:
    """
    Detects and filters reflection artifacts from water areas in images.
    
    The filter uses symmetric pattern detection via template matching,
    brightness and color similarity analysis, and morphological operations
    to identify and remove reflection regions from segmentation masks.
    """
    
    def __init__(self,
                 symmetry_threshold: float = 0.7,
                 color_similarity_threshold: float = 0.8,
                 brightness_similarity_threshold: float = 0.8,
                 min_reflection_area: int = 500,
                 template_match_threshold: float = 0.6,
                 morphology_kernel_size: int = 5):
        """
        Initialize the reflection filter with configurable parameters.
        
        Args:
            symmetry_threshold: Minimum symmetry score for reflection detection
            color_similarity_threshold: Minimum color similarity for reflections
            brightness_similarity_threshold: Minimum brightness similarity for reflections
            min_reflection_area: Minimum area for valid reflection regions
            template_match_threshold: Threshold for template matching
            morphology_kernel_size: Size of morphological operation kernels
        """
        self.symmetry_threshold = symmetry_threshold
        self.color_similarity_threshold = color_similarity_threshold
        self.brightness_similarity_threshold = brightness_similarity_threshold
        self.min_reflection_area = min_reflection_area
        self.template_match_threshold = template_match_threshold
        self.morphology_kernel_size = morphology_kernel_size
    
    def detect_reflections(self, image: np.ndarray, waterline: WaterlineBoundary) -> List[ReflectionRegion]:
        """
        Detect reflection regions in an image using the waterline boundary.
        
        Args:
            image: Input RGB image as numpy array
            waterline: Detected waterline boundary
            
        Returns:
            List of detected reflection regions
        """
        if image is None or image.size == 0:
            return []
            
        if waterline is None or not waterline.points:
            return []  # No waterline, cannot detect reflections
        
        # Step 1: Create masks for above and below waterline regions
        above_mask, below_mask = self._create_waterline_masks(image.shape, waterline)
        
        # Step 2: Extract regions above and below waterline
        above_region = cv2.bitwise_and(image, image, mask=above_mask)
        below_region = cv2.bitwise_and(image, image, mask=below_mask)
        
        # Step 3: Detect symmetric patterns using template matching
        reflection_candidates = self._detect_symmetric_patterns(above_region, below_region, waterline)
        
        # Step 4: Analyze brightness and color similarity
        validated_reflections = []
        for candidate in reflection_candidates:
            if self._validate_reflection_candidate(image, candidate, waterline):
                validated_reflections.append(candidate)
        
        # Step 5: Apply morphological operations for boundary cleanup
        cleaned_reflections = self._cleanup_reflection_boundaries(validated_reflections)
        
        return cleaned_reflections
    
    def filter_mask_reflections(self, mask: np.ndarray, reflections: List[ReflectionRegion]) -> np.ndarray:
        """
        Remove reflection regions from a segmentation mask.
        
        Args:
            mask: Input binary segmentation mask
            reflections: List of detected reflection regions
            
        Returns:
            Filtered mask with reflections removed
        """
        if mask is None or mask.size == 0:
            return mask
            
        if not reflections:
            return mask  # No reflections to filter
        
        filtered_mask = mask.copy()
        
        for reflection in reflections:
            # Remove reflection region from the mask
            filtered_mask = cv2.bitwise_and(filtered_mask, cv2.bitwise_not(reflection.mask))
        
        return filtered_mask
    
    def validate_symmetry(self, region: ReflectionRegion) -> float:
        """
        Validate the symmetry score of a reflection region.
        
        Args:
            region: Reflection region to validate
            
        Returns:
            Symmetry validation score (0.0 to 1.0)
        """
        if region is None or region.mask is None:
            return 0.0
        
        # The symmetry score is already calculated during detection
        # This method provides additional validation if needed
        return min(1.0, max(0.0, region.symmetry_score))
    
    def _create_waterline_masks(self, image_shape: Tuple[int, int], waterline: WaterlineBoundary) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create masks for regions above and below the waterline.
        
        Args:
            image_shape: Shape of the image (height, width, channels)
            waterline: Detected waterline boundary
            
        Returns:
            Tuple of (above_mask, below_mask) as binary arrays
        """
        height, width = image_shape[:2]
        
        # Initialize masks
        above_mask = np.zeros((height, width), dtype=np.uint8)
        below_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Use waterline equation to determine regions
        a, b, c = waterline.equation
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Calculate which side of the line each pixel is on
        distances = a * x_coords + b * y_coords + c
        
        if waterline.water_region_below:
            # Water is below the line
            above_mask[distances > 0] = 255  # Above waterline
            below_mask[distances <= 0] = 255  # Below waterline (water)
        else:
            # Water is above the line
            above_mask[distances <= 0] = 255  # Above waterline
            below_mask[distances > 0] = 255  # Below waterline (water)
        
        return above_mask, below_mask
    
    def _detect_symmetric_patterns(self, above_region: np.ndarray, below_region: np.ndarray, 
                                 waterline: WaterlineBoundary) -> List[ReflectionRegion]:
        """
        Detect symmetric patterns using template matching between above and below waterline regions.
        
        Args:
            above_region: Image region above the waterline
            below_region: Image region below the waterline
            waterline: Waterline boundary information
            
        Returns:
            List of reflection region candidates
        """
        candidates = []
        
        if above_region is None or below_region is None:
            return candidates
        
        # Convert to grayscale for template matching
        above_gray = cv2.cvtColor(above_region, cv2.COLOR_RGB2GRAY)
        below_gray = cv2.cvtColor(below_region, cv2.COLOR_RGB2GRAY)
        
        # Find contours in the above region to use as templates
        above_thresh = cv2.threshold(above_gray, 30, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(above_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_reflection_area:
                continue
            
            # Get bounding box for the contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract template from above region
            template = above_gray[y:y+h, x:x+w]
            if template.size == 0:
                continue
            
            # Create flipped template (reflections are vertically flipped)
            flipped_template = cv2.flip(template, 0)  # Flip vertically
            
            # Perform template matching in below region
            if below_gray.shape[0] >= flipped_template.shape[0] and below_gray.shape[1] >= flipped_template.shape[1]:
                match_result = match_template(below_gray, flipped_template)
                
                # Find matches above threshold
                match_locations = np.where(match_result >= self.template_match_threshold)
                
                for match_y, match_x in zip(match_locations[0], match_locations[1]):
                    # Create reflection region mask
                    reflection_mask = np.zeros(below_gray.shape, dtype=np.uint8)
                    reflection_mask[match_y:match_y+h, match_x:match_x+w] = 255
                    
                    # Calculate symmetry score based on template match confidence
                    symmetry_score = float(match_result[match_y, match_x])
                    
                    # Calculate color similarity
                    color_similarity = self._calculate_color_similarity(
                        above_region[y:y+h, x:x+w],
                        below_region[match_y:match_y+h, match_x:match_x+w]
                    )
                    
                    # Create reflection region
                    reflection_region = ReflectionRegion(
                        mask=reflection_mask,
                        symmetry_score=symmetry_score,
                        color_similarity=color_similarity,
                        bbox=(match_x, match_y, w, h)
                    )
                    
                    candidates.append(reflection_region)
        
        return candidates
    
    def _calculate_color_similarity(self, region1: np.ndarray, region2: np.ndarray) -> float:
        """
        Calculate color similarity between two image regions.
        
        Args:
            region1: First image region
            region2: Second image region
            
        Returns:
            Color similarity score (0.0 to 1.0)
        """
        if region1 is None or region2 is None or region1.size == 0 or region2.size == 0:
            return 0.0
        
        # Ensure regions have the same shape
        if region1.shape != region2.shape:
            # Resize to match the smaller region
            min_height = min(region1.shape[0], region2.shape[0])
            min_width = min(region1.shape[1], region2.shape[1])
            region1 = region1[:min_height, :min_width]
            region2 = region2[:min_height, :min_width]
        
        # Calculate mean colors in each channel
        mean1 = np.mean(region1, axis=(0, 1))
        mean2 = np.mean(region2, axis=(0, 1))
        
        # Calculate color distance (Euclidean distance in RGB space)
        color_distance = np.linalg.norm(mean1 - mean2)
        
        # Convert to similarity score (0 distance = 1.0 similarity)
        max_distance = np.sqrt(3 * 255**2)  # Maximum possible RGB distance
        similarity = 1.0 - (color_distance / max_distance)
        
        return max(0.0, min(1.0, similarity))
    
    def _calculate_brightness_similarity(self, region1: np.ndarray, region2: np.ndarray) -> float:
        """
        Calculate brightness similarity between two image regions.
        
        Args:
            region1: First image region
            region2: Second image region
            
        Returns:
            Brightness similarity score (0.0 to 1.0)
        """
        if region1 is None or region2 is None or region1.size == 0 or region2.size == 0:
            return 0.0
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(region1, cv2.COLOR_RGB2GRAY) if len(region1.shape) == 3 else region1
        gray2 = cv2.cvtColor(region2, cv2.COLOR_RGB2GRAY) if len(region2.shape) == 3 else region2
        
        # Ensure regions have the same shape
        if gray1.shape != gray2.shape:
            min_height = min(gray1.shape[0], gray2.shape[0])
            min_width = min(gray1.shape[1], gray2.shape[1])
            gray1 = gray1[:min_height, :min_width]
            gray2 = gray2[:min_height, :min_width]
        
        # Calculate mean brightness
        mean1 = np.mean(gray1)
        mean2 = np.mean(gray2)
        
        # Calculate brightness difference
        brightness_diff = abs(mean1 - mean2)
        
        # Convert to similarity score
        similarity = 1.0 - (brightness_diff / 255.0)
        
        return max(0.0, min(1.0, similarity))
    
    def _validate_reflection_candidate(self, image: np.ndarray, candidate: ReflectionRegion, 
                                     waterline: WaterlineBoundary) -> bool:
        """
        Validate a reflection candidate using brightness and color similarity analysis.
        
        Args:
            image: Original input image
            candidate: Reflection region candidate
            waterline: Waterline boundary
            
        Returns:
            True if candidate is a valid reflection, False otherwise
        """
        if candidate is None or candidate.mask is None:
            return False
        
        # Check if symmetry score meets threshold
        if candidate.symmetry_score < self.symmetry_threshold:
            return False
        
        # Check if color similarity meets threshold
        if candidate.color_similarity < self.color_similarity_threshold:
            return False
        
        # Extract the reflection region from the image
        reflection_region = cv2.bitwise_and(image, image, mask=candidate.mask)
        
        # Find corresponding region above waterline for brightness comparison
        x, y, w, h = candidate.bbox
        
        # Calculate the mirrored position above the waterline
        # This is a simplified approach - in practice, you might need more sophisticated mirroring
        waterline_y = int(np.mean([point[1] for point in waterline.points]))
        mirror_y = waterline_y - (y + h - waterline_y)
        
        if mirror_y >= 0 and mirror_y + h < image.shape[0]:
            above_region = image[mirror_y:mirror_y+h, x:x+w]
            below_region = image[y:y+h, x:x+w]
            
            # Calculate brightness similarity
            brightness_similarity = self._calculate_brightness_similarity(above_region, below_region)
            
            if brightness_similarity < self.brightness_similarity_threshold:
                return False
        
        # Check minimum area requirement
        reflection_area = np.sum(candidate.mask > 0)
        if reflection_area < self.min_reflection_area:
            return False
        
        return True
    
    def _cleanup_reflection_boundaries(self, reflections: List[ReflectionRegion]) -> List[ReflectionRegion]:
        """
        Apply morphological operations to clean up reflection boundaries.
        
        Args:
            reflections: List of detected reflection regions
            
        Returns:
            List of reflection regions with cleaned boundaries
        """
        cleaned_reflections = []
        
        for reflection in reflections:
            if reflection.mask is None:
                continue
            
            # Apply morphological opening to remove small noise
            kernel = disk(self.morphology_kernel_size)
            cleaned_mask = binary_opening(reflection.mask > 0, kernel)
            
            # Apply morphological closing to fill small gaps
            cleaned_mask = binary_closing(cleaned_mask, kernel)
            
            # Convert back to uint8
            cleaned_mask = (cleaned_mask * 255).astype(np.uint8)
            
            # Update bounding box after morphological operations
            contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Use the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Create updated reflection region
                cleaned_reflection = ReflectionRegion(
                    mask=cleaned_mask,
                    symmetry_score=reflection.symmetry_score,
                    color_similarity=reflection.color_similarity,
                    bbox=(x, y, w, h)
                )
                
                cleaned_reflections.append(cleaned_reflection)
        
        return cleaned_reflections
    
    def create_reflection_exclusion_mask(self, image_shape: Tuple[int, int], 
                                       reflections: List[ReflectionRegion]) -> np.ndarray:
        """
        Create a mask that excludes reflection regions.
        
        Args:
            image_shape: Shape of the image (height, width)
            reflections: List of detected reflection regions
            
        Returns:
            Binary mask where non-reflection regions are white (255) and reflection regions are black (0)
        """
        height, width = image_shape[:2]
        exclusion_mask = np.ones((height, width), dtype=np.uint8) * 255
        
        for reflection in reflections:
            if reflection.mask is not None:
                # Set reflection regions to black (exclude them)
                exclusion_mask = cv2.bitwise_and(exclusion_mask, cv2.bitwise_not(reflection.mask))
        
        return exclusion_mask