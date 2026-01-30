"""
Quality validation and confidence metrics for manatee segmentation.

This module provides functionality to validate segmentation quality, calculate confidence
metrics, and flag results that may need manual review based on coherence and quality thresholds.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import ndimage
from skimage import measure, morphology
from skimage.filters import gaussian

from src.models.data_models import SegmentationMask, ProcessingResult


class QualityValidator:
    """
    Validates segmentation quality and provides confidence metrics.
    
    This class analyzes segmented regions for coherence, calculates quality scores,
    and provides threshold-based flagging for manual review.
    """
    
    def __init__(self, coherence_threshold: float = 0.7, 
                 confidence_threshold: float = 0.6,
                 min_area_threshold: int = 1000):
        """
        Initialize the quality validator.
        
        Args:
            coherence_threshold: Minimum coherence score for acceptable segmentation
            confidence_threshold: Minimum confidence score for automatic approval
            min_area_threshold: Minimum area in pixels for valid segmentation
        """
        self.coherence_threshold = coherence_threshold
        self.confidence_threshold = confidence_threshold
        self.min_area_threshold = min_area_threshold
        self.logger = logging.getLogger(__name__)
    
    def validate_coherence(self, mask: np.ndarray) -> float:
        """
        Validate that segmented regions form coherent shapes.
        
        Args:
            mask: Binary segmentation mask
            
        Returns:
            Coherence score between 0.0 and 1.0 (higher is better)
            
        Raises:
            ValueError: If mask is invalid
        """
        if mask is None:
            raise ValueError("Mask cannot be None")
        
        if mask.size == 0:
            return 0.0
        
        try:
            # Convert to binary if needed
            binary_mask = mask > 0 if mask.dtype != bool else mask
            
            if not np.any(binary_mask):
                return 0.0  # No segmented regions
            
            # Label connected components
            labeled_mask = measure.label(binary_mask)
            regions = measure.regionprops(labeled_mask)
            
            if not regions:
                return 0.0
            
            # Calculate coherence metrics for each region
            coherence_scores = []
            
            for region in regions:
                if region.area < self.min_area_threshold:
                    continue  # Skip very small regions
                
                # Metric 1: Solidity (area / convex_hull_area)
                solidity = region.solidity
                
                # Metric 2: Extent (area / bounding_box_area)
                extent = region.extent
                
                # Metric 3: Eccentricity (measure of elongation)
                eccentricity = region.eccentricity
                eccentricity_score = 1.0 - min(eccentricity, 0.9)  # Prefer less elongated shapes
                
                # Metric 4: Compactness (perimeter^2 / area)
                perimeter = region.perimeter
                if perimeter > 0:
                    compactness = (4 * np.pi * region.area) / (perimeter ** 2)
                    compactness_score = min(compactness, 1.0)
                else:
                    compactness_score = 0.0
                
                # Combine metrics with weights
                region_coherence = (
                    0.3 * solidity +
                    0.2 * extent +
                    0.2 * eccentricity_score +
                    0.3 * compactness_score
                )
                
                coherence_scores.append(region_coherence)
            
            if not coherence_scores:
                return 0.0
            
            # Return weighted average by area
            total_area = sum(region.area for region in regions if region.area >= self.min_area_threshold)
            if total_area == 0:
                return 0.0
            
            weighted_coherence = sum(
                score * region.area for score, region in zip(coherence_scores, regions)
                if region.area >= self.min_area_threshold
            ) / total_area
            
            self.logger.debug(f"Calculated coherence score: {weighted_coherence:.3f}")
            return min(max(weighted_coherence, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating coherence: {e}")
            return 0.0
    
    def calculate_confidence_score(self, segmentation: SegmentationMask, 
                                 mask: np.ndarray,
                                 original_image: Optional[np.ndarray] = None) -> float:
        """
        Calculate overall confidence score for segmentation quality.
        
        Args:
            segmentation: Original segmentation mask with metadata
            mask: Final processed binary mask
            original_image: Optional original image for additional analysis
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if segmentation is None or mask is None:
            return 0.0
        
        try:
            # Start with base confidence from segmentation
            base_confidence = segmentation.confidence
            stability_score = segmentation.stability_score
            
            # Calculate coherence score
            coherence_score = self.validate_coherence(mask)
            
            # Calculate area consistency score
            area_score = self._calculate_area_score(segmentation, mask)
            
            # Calculate edge quality score
            edge_score = self._calculate_edge_quality(mask)
            
            # Optional: Calculate color consistency if original image provided
            color_score = 1.0
            if original_image is not None:
                color_score = self._calculate_color_consistency(mask, original_image)
            
            # Combine scores with weights
            final_confidence = (
                0.25 * base_confidence +
                0.15 * stability_score +
                0.25 * coherence_score +
                0.15 * area_score +
                0.10 * edge_score +
                0.10 * color_score
            )
            
            self.logger.debug(f"Confidence components - base: {base_confidence:.3f}, "
                            f"stability: {stability_score:.3f}, coherence: {coherence_score:.3f}, "
                            f"area: {area_score:.3f}, edge: {edge_score:.3f}, color: {color_score:.3f}")
            
            return min(max(final_confidence, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence score: {e}")
            return 0.0
    
    def flag_for_manual_review(self, confidence_score: float, 
                             coherence_score: float,
                             mask: np.ndarray) -> Dict[str, any]:
        """
        Determine if segmentation should be flagged for manual review.
        
        Args:
            confidence_score: Overall confidence score
            coherence_score: Coherence validation score
            mask: Binary segmentation mask
            
        Returns:
            Dictionary with flagging decision and reasons
        """
        flags = {
            "needs_review": False,
            "reasons": [],
            "severity": "none",  # none, low, medium, high
            "confidence_score": confidence_score,
            "coherence_score": coherence_score
        }
        
        try:
            # Check confidence threshold
            if confidence_score < self.confidence_threshold:
                flags["needs_review"] = True
                flags["reasons"].append(f"Low confidence score: {confidence_score:.3f}")
            
            # Check coherence threshold
            if coherence_score < self.coherence_threshold:
                flags["needs_review"] = True
                flags["reasons"].append(f"Low coherence score: {coherence_score:.3f}")
            
            # Check minimum area
            if mask is not None:
                mask_area = np.sum(mask > 0)
                if mask_area < self.min_area_threshold:
                    flags["needs_review"] = True
                    flags["reasons"].append(f"Segmented area too small: {mask_area} pixels")
                
                # Check for fragmentation
                labeled_mask = measure.label(mask > 0)
                num_components = len(np.unique(labeled_mask)) - 1  # Subtract background
                if num_components > 3:
                    flags["needs_review"] = True
                    flags["reasons"].append(f"Too many disconnected components: {num_components}")
            
            # Determine severity
            if flags["needs_review"]:
                if confidence_score < 0.3 or coherence_score < 0.3:
                    flags["severity"] = "high"
                elif confidence_score < 0.5 or coherence_score < 0.5:
                    flags["severity"] = "medium"
                else:
                    flags["severity"] = "low"
            
            return flags
            
        except Exception as e:
            self.logger.error(f"Error in flagging logic: {e}")
            flags["needs_review"] = True
            flags["reasons"].append(f"Error in quality assessment: {e}")
            flags["severity"] = "high"
            return flags
    
    def generate_quality_report(self, segmentation: SegmentationMask,
                              final_mask: np.ndarray,
                              original_image: Optional[np.ndarray] = None) -> Dict[str, any]:
        """
        Generate a comprehensive quality report for the segmentation.
        
        Args:
            segmentation: Original segmentation mask with metadata
            final_mask: Final processed binary mask
            original_image: Optional original image
            
        Returns:
            Dictionary with comprehensive quality metrics and recommendations
        """
        try:
            # Calculate all quality metrics
            coherence_score = self.validate_coherence(final_mask)
            confidence_score = self.calculate_confidence_score(segmentation, final_mask, original_image)
            flags = self.flag_for_manual_review(confidence_score, coherence_score, final_mask)
            
            # Calculate additional statistics
            mask_stats = self._get_mask_statistics(final_mask)
            
            report = {
                "overall_quality": "excellent" if confidence_score > 0.8 else
                                 "good" if confidence_score > 0.6 else
                                 "fair" if confidence_score > 0.4 else "poor",
                "confidence_score": confidence_score,
                "coherence_score": coherence_score,
                "segmentation_confidence": segmentation.confidence,
                "stability_score": segmentation.stability_score,
                "mask_statistics": mask_stats,
                "quality_flags": flags,
                "recommendations": self._generate_recommendations(flags, confidence_score, coherence_score)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating quality report: {e}")
            return {
                "overall_quality": "error",
                "error": str(e),
                "confidence_score": 0.0,
                "coherence_score": 0.0
            }
    
    def _calculate_area_score(self, segmentation: SegmentationMask, mask: np.ndarray) -> float:
        """Calculate score based on area consistency between original and final mask."""
        try:
            original_area = segmentation.area
            final_area = np.sum(mask > 0)
            
            if original_area == 0:
                return 0.0 if final_area > 0 else 1.0
            
            # Calculate area retention ratio
            retention_ratio = final_area / original_area
            
            # Penalize significant area loss or gain
            if 0.7 <= retention_ratio <= 1.0:
                return 1.0
            elif 0.5 <= retention_ratio < 0.7:
                return 0.8
            elif 0.3 <= retention_ratio < 0.5:
                return 0.6
            else:
                return 0.3
                
        except Exception:
            return 0.5  # Neutral score on error
    
    def _calculate_edge_quality(self, mask: np.ndarray) -> float:
        """Calculate edge quality score based on smoothness and continuity."""
        try:
            if mask is None or not np.any(mask):
                return 0.0
            
            # Apply Gaussian smoothing and calculate edge strength
            smoothed = gaussian(mask.astype(float), sigma=1.0)
            edges = np.gradient(smoothed)
            edge_magnitude = np.sqrt(edges[0]**2 + edges[1]**2)
            
            # Calculate edge smoothness (lower variance is better)
            edge_variance = np.var(edge_magnitude[edge_magnitude > 0.1])
            smoothness_score = 1.0 / (1.0 + edge_variance)
            
            return min(max(smoothness_score, 0.0), 1.0)
            
        except Exception:
            return 0.5  # Neutral score on error
    
    def _calculate_color_consistency(self, mask: np.ndarray, image: np.ndarray) -> float:
        """Calculate color consistency within segmented regions."""
        try:
            if mask is None or image is None or not np.any(mask):
                return 0.5
            
            # Extract pixels within the mask
            mask_pixels = image[mask > 0]
            
            if len(mask_pixels) == 0:
                return 0.0
            
            # Calculate color variance (lower is better for consistency)
            color_variance = np.mean(np.var(mask_pixels, axis=0))
            consistency_score = 1.0 / (1.0 + color_variance / 100.0)
            
            return min(max(consistency_score, 0.0), 1.0)
            
        except Exception:
            return 0.5  # Neutral score on error
    
    def _get_mask_statistics(self, mask: np.ndarray) -> Dict[str, any]:
        """Get detailed statistics about the mask."""
        try:
            if mask is None:
                return {"error": "Mask is None"}
            
            binary_mask = mask > 0
            labeled_mask = measure.label(binary_mask)
            regions = measure.regionprops(labeled_mask)
            
            stats = {
                "total_pixels": mask.size,
                "segmented_pixels": int(np.sum(binary_mask)),
                "segmentation_percentage": float(np.sum(binary_mask) / mask.size * 100),
                "num_components": len(regions),
                "largest_component_area": max([r.area for r in regions]) if regions else 0,
                "average_component_area": np.mean([r.area for r in regions]) if regions else 0
            }
            
            return stats
            
        except Exception as e:
            return {"error": f"Failed to calculate statistics: {e}"}
    
    def _generate_recommendations(self, flags: Dict, confidence_score: float, 
                                coherence_score: float) -> List[str]:
        """Generate recommendations based on quality assessment."""
        recommendations = []
        
        if flags["needs_review"]:
            if confidence_score < 0.5:
                recommendations.append("Consider adjusting SAM parameters or using different prompts")
            
            if coherence_score < 0.5:
                recommendations.append("Apply morphological operations to improve shape coherence")
            
            if "Too many disconnected components" in flags["reasons"]:
                recommendations.append("Consider post-processing to merge nearby components")
            
            if "Segmented area too small" in flags["reasons"]:
                recommendations.append("Verify image contains a visible manatee above waterline")
        
        if not recommendations:
            recommendations.append("Segmentation quality is acceptable for automated processing")
        
        return recommendations