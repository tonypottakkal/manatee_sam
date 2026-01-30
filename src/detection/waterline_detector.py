"""
Waterline detection module for manatee segmentation system.

This module implements waterline detection using HSV color space analysis,
Canny edge detection, Hough line transform, and texture analysis using LBP features.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from skimage.feature import local_binary_pattern
from scipy import ndimage

# Use absolute imports to avoid relative import issues
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_models import WaterlineBoundary, WaterRegion, Vector3D


class WaterlineDetector:
    """
    Detects waterline boundaries in images using multiple computer vision techniques.
    
    The detector combines HSV color space analysis for water detection,
    Canny edge detection and Hough line transform for boundary detection,
    and texture analysis using Local Binary Pattern (LBP) features for validation.
    """
    
    def __init__(self, 
                 water_hue_range: Tuple[int, int] = (90, 130),  # Blue-green water hues
                 water_saturation_min: float = 0.3,
                 edge_threshold_low: int = 50,
                 edge_threshold_high: int = 150,
                 hough_threshold: int = 100,
                 min_line_length: int = 50,
                 max_line_gap: int = 10,
                 lbp_radius: int = 3,
                 lbp_n_points: int = 24):
        """
        Initialize the waterline detector with configurable parameters.
        
        Args:
            water_hue_range: HSV hue range for water detection (degrees)
            water_saturation_min: Minimum saturation for water regions
            edge_threshold_low: Lower threshold for Canny edge detection
            edge_threshold_high: Upper threshold for Canny edge detection
            hough_threshold: Threshold for Hough line detection
            min_line_length: Minimum line length for Hough transform
            max_line_gap: Maximum gap between line segments
            lbp_radius: Radius for LBP texture analysis
            lbp_n_points: Number of points for LBP texture analysis
        """
        self.water_hue_range = water_hue_range
        self.water_saturation_min = water_saturation_min
        self.edge_threshold_low = edge_threshold_low
        self.edge_threshold_high = edge_threshold_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.lbp_radius = lbp_radius
        self.lbp_n_points = lbp_n_points
    
    def detect_waterline(self, image: np.ndarray) -> Optional[WaterlineBoundary]:
        """
        Detect the waterline boundary in an image.
        
        Args:
            image: Input RGB image as numpy array
            
        Returns:
            WaterlineBoundary object if waterline is detected, None otherwise
        """
        if image is None or image.size == 0:
            return None
            
        # Step 1: Analyze water regions using HSV color space
        water_regions = self.analyze_water_regions(image)
        if not water_regions:
            return None
            
        # Step 2: Detect edges using Canny edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, self.edge_threshold_low, self.edge_threshold_high)
        
        # Step 3: Apply Hough line transform to find dominant horizontal lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                               threshold=self.hough_threshold,
                               minLineLength=self.min_line_length,
                               maxLineGap=self.max_line_gap)
        
        if lines is None or len(lines) == 0:
            return None
            
        # Step 4: Filter for horizontal lines and find the most prominent waterline
        horizontal_lines = self._filter_horizontal_lines(lines, image.shape)
        if not horizontal_lines:
            return None
            
        # Step 5: Validate waterline candidates using texture analysis
        best_waterline = self._validate_waterline_candidates(
            image, horizontal_lines, water_regions)
        
        return best_waterline
    
    def analyze_water_regions(self, image: np.ndarray) -> List[WaterRegion]:
        """
        Analyze image to identify water regions using HSV color space.
        
        Args:
            image: Input RGB image as numpy array
            
        Returns:
            List of detected water regions
        """
        if image is None or image.size == 0:
            return []
            
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Create mask for water-like colors (blue-green hues with sufficient saturation)
        hue_min, hue_max = self.water_hue_range
        lower_water = np.array([hue_min, int(self.water_saturation_min * 255), 50])
        upper_water = np.array([hue_max, 255, 255])
        
        water_mask = cv2.inRange(hsv, lower_water, upper_water)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
        
        # Find connected components (water regions)
        contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        water_regions = []
        for contour in contours:
            # Filter out small regions
            area = cv2.contourArea(contour)
            if area < 1000:  # Minimum area threshold
                continue
                
            # Create region mask
            region_mask = np.zeros(water_mask.shape, dtype=np.uint8)
            cv2.fillPoly(region_mask, [contour], 255)
            
            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate average HSV values in the region
            region_hsv = hsv[region_mask > 0]
            avg_hue = np.mean(region_hsv[:, 0]) * 2  # Convert to degrees
            avg_saturation = np.mean(region_hsv[:, 1]) / 255.0
            
            # Calculate confidence based on color consistency
            hue_std = np.std(region_hsv[:, 0])
            sat_std = np.std(region_hsv[:, 1])
            confidence = max(0.0, 1.0 - (hue_std + sat_std) / 100.0)
            
            water_region = WaterRegion(
                mask=region_mask,
                bbox=(x, y, w, h),
                area=int(area),
                confidence=confidence,
                avg_hue=avg_hue,
                avg_saturation=avg_saturation
            )
            
            water_regions.append(water_region)
        
        return water_regions
    
    def estimate_surface_normal(self, boundary: WaterlineBoundary) -> Vector3D:
        """
        Estimate the surface normal vector from a waterline boundary.
        
        Args:
            boundary: Detected waterline boundary
            
        Returns:
            Normalized 3D vector representing the surface normal
        """
        if not boundary or not boundary.points or len(boundary.points) < 2:
            return Vector3D(0, 0, 1)  # Default upward normal
            
        # Calculate the direction vector of the waterline
        points = np.array(boundary.points)
        if len(points) < 2:
            return Vector3D(0, 0, 1)
            
        # Use first and last points to determine line direction
        start_point = points[0]
        end_point = points[-1]
        
        # Calculate 2D direction vector
        direction_2d = end_point - start_point
        direction_2d = direction_2d / np.linalg.norm(direction_2d)
        
        # The surface normal is perpendicular to the waterline direction
        # Assuming the waterline is horizontal, the normal points upward
        normal_2d = np.array([-direction_2d[1], direction_2d[0]])
        
        # Convert to 3D (assuming z-axis points up from the image plane)
        normal_3d = Vector3D(normal_2d[0], normal_2d[1], 0.5)
        
        return normal_3d.normalize()
    
    def _filter_horizontal_lines(self, lines: np.ndarray, image_shape: Tuple[int, int]) -> List[Tuple[np.ndarray, float]]:
        """
        Filter detected lines to find horizontal waterline candidates.
        
        Args:
            lines: Lines detected by Hough transform
            image_shape: Shape of the input image (height, width)
            
        Returns:
            List of (line, score) tuples for horizontal line candidates
        """
        horizontal_lines = []
        height, width = image_shape[:2]
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line angle
            if x2 - x1 == 0:
                angle = 90  # Vertical line
            else:
                angle = abs(np.degrees(np.arctan((y2 - y1) / (x2 - x1))))
            
            # Filter for nearly horizontal lines (within 15 degrees of horizontal)
            if angle <= 15 or angle >= 165:
                # Calculate line length
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Calculate line position (prefer lines in lower half of image)
                avg_y = (y1 + y2) / 2
                position_score = avg_y / height  # Higher score for lower lines
                
                # Calculate line span (prefer lines that span more of the image width)
                span = abs(x2 - x1) / width
                
                # Combined score
                score = length * position_score * span
                
                horizontal_lines.append((line[0], score))
        
        # Sort by score (descending)
        horizontal_lines.sort(key=lambda x: x[1], reverse=True)
        
        return horizontal_lines
    
    def _validate_waterline_candidates(self, 
                                     image: np.ndarray, 
                                     line_candidates: List[Tuple[np.ndarray, float]], 
                                     water_regions: List[WaterRegion]) -> Optional[WaterlineBoundary]:
        """
        Validate waterline candidates using texture analysis and water region overlap.
        
        Args:
            image: Input RGB image
            line_candidates: List of (line, score) tuples
            water_regions: Detected water regions
            
        Returns:
            Best validated waterline boundary or None
        """
        if not line_candidates or not water_regions:
            return None
            
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        best_waterline = None
        best_score = 0.0
        
        for line, line_score in line_candidates[:5]:  # Check top 5 candidates
            x1, y1, x2, y2 = line
            
            # Create line points
            num_points = max(10, abs(x2 - x1) // 5)
            if x2 != x1:
                x_points = np.linspace(x1, x2, num_points, dtype=int)
                y_points = np.linspace(y1, y2, num_points, dtype=int)
            else:
                x_points = np.full(num_points, x1, dtype=int)
                y_points = np.linspace(y1, y2, num_points, dtype=int)
            
            points = [(int(x), int(y)) for x, y in zip(x_points, y_points)]
            
            # Calculate line equation (ax + by + c = 0)
            if x2 - x1 == 0:
                # Vertical line: x = x1 -> x - x1 = 0
                a, b, c = 1, 0, -x1
            else:
                # y - y1 = m(x - x1) -> mx - y + (y1 - mx1) = 0
                m = (y2 - y1) / (x2 - x1)
                a, b, c = m, -1, y1 - m * x1
            
            # Normalize the equation
            norm = np.sqrt(a**2 + b**2)
            if norm > 0:
                a, b, c = a/norm, b/norm, c/norm
            
            # Validate using texture analysis
            texture_score = self._analyze_waterline_texture(gray, points)
            
            # Validate using water region overlap
            water_overlap_score = self._calculate_water_overlap(points, water_regions, image.shape)
            
            # Combined confidence score
            confidence = (line_score * 0.4 + texture_score * 0.3 + water_overlap_score * 0.3)
            confidence = min(1.0, confidence / 100.0)  # Normalize to [0, 1]
            
            if confidence > best_score:
                best_score = confidence
                best_waterline = WaterlineBoundary(
                    points=points,
                    equation=(a, b, c),
                    confidence=confidence,
                    water_region_below=True  # Assume water is below the line
                )
        
        return best_waterline
    
    def _analyze_waterline_texture(self, gray_image: np.ndarray, line_points: List[Tuple[int, int]]) -> float:
        """
        Analyze texture around waterline using Local Binary Pattern (LBP).
        
        Args:
            gray_image: Grayscale image
            line_points: Points along the waterline
            
        Returns:
            Texture consistency score
        """
        if not line_points:
            return 0.0
            
        height, width = gray_image.shape
        
        # Sample regions above and below the line
        above_textures = []
        below_textures = []
        
        for x, y in line_points:
            if not (0 <= x < width and 0 <= y < height):
                continue
                
            # Sample region above the line
            y_above = max(0, y - 20)
            if y_above < y:
                region_above = gray_image[y_above:y, max(0, x-10):min(width, x+10)]
                if region_above.size > 0:
                    lbp_above = local_binary_pattern(region_above, 
                                                   self.lbp_n_points, 
                                                   self.lbp_radius, 
                                                   method='uniform')
                    above_textures.append(np.mean(lbp_above))
            
            # Sample region below the line
            y_below = min(height, y + 20)
            if y_below > y:
                region_below = gray_image[y:y_below, max(0, x-10):min(width, x+10)]
                if region_below.size > 0:
                    lbp_below = local_binary_pattern(region_below, 
                                                   self.lbp_n_points, 
                                                   self.lbp_radius, 
                                                   method='uniform')
                    below_textures.append(np.mean(lbp_below))
        
        if not above_textures or not below_textures:
            return 0.0
            
        # Calculate texture difference (waterlines should show texture discontinuity)
        above_mean = np.mean(above_textures)
        below_mean = np.mean(below_textures)
        texture_difference = abs(above_mean - below_mean)
        
        # Higher difference indicates a stronger boundary
        return min(100.0, texture_difference * 10.0)
    
    def _calculate_water_overlap(self, 
                               line_points: List[Tuple[int, int]], 
                               water_regions: List[WaterRegion], 
                               image_shape: Tuple[int, int]) -> float:
        """
        Calculate how well the waterline aligns with detected water regions.
        
        Args:
            line_points: Points along the waterline
            water_regions: Detected water regions
            image_shape: Shape of the image
            
        Returns:
            Water overlap score
        """
        if not line_points or not water_regions:
            return 0.0
            
        height, width = image_shape[:2]
        
        # Create a mask for the waterline
        line_mask = np.zeros((height, width), dtype=np.uint8)
        for x, y in line_points:
            if 0 <= x < width and 0 <= y < height:
                # Draw a thick line to account for slight misalignments
                cv2.circle(line_mask, (x, y), 3, 255, -1)
        
        # Check overlap with water regions
        total_overlap = 0
        total_water_area = 0
        
        for water_region in water_regions:
            # Dilate water region mask to account for boundary proximity
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            dilated_water = cv2.dilate(water_region.mask, kernel, iterations=1)
            
            # Calculate overlap
            overlap = cv2.bitwise_and(line_mask, dilated_water)
            overlap_area = np.sum(overlap > 0)
            
            total_overlap += overlap_area
            total_water_area += np.sum(water_region.mask > 0)
        
        if total_water_area == 0:
            return 0.0
            
        # Score based on how much of the waterline is near water regions
        line_area = np.sum(line_mask > 0)
        if line_area == 0:
            return 0.0
            
        overlap_ratio = total_overlap / line_area
        return min(100.0, overlap_ratio * 100.0)
    
    def filter_mask_by_waterline(self, mask: np.ndarray, waterline: WaterlineBoundary) -> np.ndarray:
        """
        Filter a segmentation mask to exclude underwater portions based on waterline boundary.
        
        Args:
            mask: Input binary segmentation mask
            waterline: Detected waterline boundary
            
        Returns:
            Filtered mask with underwater portions removed
        """
        if mask is None or mask.size == 0:
            return mask
            
        if waterline is None or not waterline.points:
            return mask  # No waterline detected, return original mask
            
        height, width = mask.shape[:2]
        
        # Create a mask for the region above the waterline
        above_waterline_mask = np.ones((height, width), dtype=np.uint8) * 255
        
        # Use the waterline equation to determine which pixels are above/below
        a, b, c = waterline.equation
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Calculate distance from each pixel to the waterline
        # For line equation ax + by + c = 0, distance = (ax + by + c) / sqrt(a^2 + b^2)
        distances = a * x_coords + b * y_coords + c
        
        # If water_region_below is True, keep pixels where distance > 0 (above the line)
        # If water_region_below is False, keep pixels where distance < 0 (below the line)
        if waterline.water_region_below:
            above_waterline_mask[distances <= 0] = 0
        else:
            above_waterline_mask[distances >= 0] = 0
        
        # Apply the waterline filter to the input mask
        filtered_mask = cv2.bitwise_and(mask, above_waterline_mask)
        
        return filtered_mask
    
    def create_underwater_exclusion_mask(self, image_shape: Tuple[int, int], waterline: WaterlineBoundary) -> np.ndarray:
        """
        Create a mask that excludes underwater regions based on waterline boundary.
        
        Args:
            image_shape: Shape of the image (height, width)
            waterline: Detected waterline boundary
            
        Returns:
            Binary mask where above-water regions are white (255) and underwater regions are black (0)
        """
        height, width = image_shape[:2]
        
        if waterline is None or not waterline.points:
            # No waterline detected, return full mask (assume everything is above water)
            return np.ones((height, width), dtype=np.uint8) * 255
        
        # Create exclusion mask
        exclusion_mask = np.ones((height, width), dtype=np.uint8) * 255
        
        # Use the waterline equation to determine which pixels are above/below
        a, b, c = waterline.equation
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Calculate which side of the line each pixel is on
        distances = a * x_coords + b * y_coords + c
        
        # Exclude underwater regions
        if waterline.water_region_below:
            exclusion_mask[distances <= 0] = 0  # Below the line is underwater
        else:
            exclusion_mask[distances >= 0] = 0  # Above the line is underwater
        
        return exclusion_mask