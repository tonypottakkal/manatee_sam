"""
SAM-based segmentation engine for manatee body detection.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    import torch
except ImportError as e:
    logging.warning(f"SAM dependencies not available: {e}")
    # Define placeholder classes for development
    class SamAutomaticMaskGenerator:
        def __init__(self, *args, **kwargs):
            pass
        def generate(self, image):
            return []
    
    sam_model_registry = {}
    
    class torch:
        class cuda:
            @staticmethod
            def is_available():
                return False

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.data_models import SegmentationMask
from src.segmentation.mask_processor import MaskProcessor


class SAMSegmentationEngine:
    """
    Segmentation engine using Meta's Segment Anything Model (SAM).
    
    This class handles:
    - SAM model initialization with ViT-H backbone
    - Automatic mask generation from input images
    - Mask filtering by size and coherence validation
    - Primary subject selection for manatee detection
    """
    
    def __init__(self, model_type: str = "vit_h", checkpoint_path: Optional[str] = None,
                 enable_integrated_filtering: bool = False):
        """
        Initialize the SAM segmentation engine.
        
        Args:
            model_type: SAM model type (vit_h, vit_l, vit_b)
            checkpoint_path: Path to SAM model checkpoint file
            enable_integrated_filtering: Whether to enable integrated waterline and reflection filtering
        """
        self.model_type = model_type
        self.enable_integrated_filtering = enable_integrated_filtering
        
        # Use local checkpoint if available and none specified
        if checkpoint_path is None:
            local_checkpoint = f"sam_checkpoints/sam_vit_{model_type[4:]}_4b8939.pth"
            if Path(local_checkpoint).exists():
                self.checkpoint_path = local_checkpoint
            else:
                self.checkpoint_path = None
        else:
            self.checkpoint_path = checkpoint_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam_model = None
        self.mask_generator = None
        
        # Initialize mask processor for integrated filtering
        self.mask_processor = MaskProcessor() if enable_integrated_filtering else None
        
        # Configuration parameters for mask filtering
        self.min_mask_area = 1000  # Minimum area for valid masks
        self.max_mask_area_ratio = 0.8  # Maximum ratio of image area
        self.stability_score_threshold = 0.7  # Minimum stability score
        self.confidence_threshold = 0.5  # Minimum confidence score
        
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the SAM model and mask generator."""
        try:
            if self.model_type not in sam_model_registry:
                available_models = list(sam_model_registry.keys())
                raise ValueError(f"Model type '{self.model_type}' not available. "
                               f"Available models: {available_models}")
            
            # Load SAM model
            if self.checkpoint_path and Path(self.checkpoint_path).exists():
                self.sam_model = sam_model_registry[self.model_type](
                    checkpoint=self.checkpoint_path
                )
            else:
                # For development, create a placeholder model
                logging.warning("SAM checkpoint not found, using placeholder model")
                self.sam_model = None
            
            if self.sam_model:
                self.sam_model.to(device=self.device)
                
                # Initialize automatic mask generator
                self.mask_generator = SamAutomaticMaskGenerator(
                    model=self.sam_model,
                    points_per_side=32,
                    pred_iou_thresh=0.88,
                    stability_score_thresh=self.stability_score_threshold,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=self.min_mask_area,
                )
            else:
                # Create a placeholder mask generator for development
                self.mask_generator = None
                
        except Exception as e:
            logging.error(f"Failed to initialize SAM model: {e}")
            self.sam_model = None
            self.mask_generator = None
    
    def generate_masks(self, image: np.ndarray) -> List[SegmentationMask]:
        """
        Generate segmentation masks from input image using SAM.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            List of SegmentationMask objects with metadata
        """
        if image is None or image.size == 0:
            return []
        
        if len(image.shape) != 3 or image.shape[2] not in [3, 4]:
            raise ValueError("Input image must be RGB or RGBA format (H, W, C)")
        
        try:
            # Convert RGBA to RGB if needed
            if image.shape[2] == 4:
                image = image[:, :, :3]
            
            # Generate masks using SAM
            if self.mask_generator is None:
                logging.warning("SAM mask generator not initialized - returning empty mask list")
                return []
            
            sam_masks = self.mask_generator.generate(image)
            
            # Convert SAM masks to our SegmentationMask format
            segmentation_masks = []
            for sam_mask in sam_masks:
                try:
                    mask = sam_mask.get('segmentation', np.array([]))
                    if mask.size == 0:
                        continue
                    
                    # Extract bounding box
                    bbox = sam_mask.get('bbox', [0, 0, 0, 0])
                    if len(bbox) != 4:
                        continue
                    
                    # Calculate area
                    area = int(sam_mask.get('area', np.sum(mask)))
                    
                    # Extract confidence and stability scores
                    confidence = float(sam_mask.get('predicted_iou', 0.0))
                    stability_score = float(sam_mask.get('stability_score', 0.0))
                    
                    # Create SegmentationMask object
                    seg_mask = SegmentationMask(
                        mask=mask.astype(bool),
                        bbox=tuple(bbox),
                        area=area,
                        confidence=confidence,
                        stability_score=stability_score
                    )
                    
                    # Validate the mask
                    if seg_mask.validate():
                        segmentation_masks.append(seg_mask)
                    
                except Exception as e:
                    logging.warning(f"Failed to process SAM mask: {e}")
                    continue
            
            return segmentation_masks
            
        except Exception as e:
            logging.error(f"Failed to generate masks: {e}")
            return []
    
    def filter_by_size(self, masks: List[SegmentationMask], 
                      image_shape: Optional[Tuple[int, int]] = None) -> List[SegmentationMask]:
        """
        Filter masks by size constraints to focus on large objects.
        
        Args:
            masks: List of segmentation masks to filter
            image_shape: Optional image shape (height, width) for ratio calculations
            
        Returns:
            Filtered list of masks meeting size criteria
        """
        if not masks:
            return []
        
        filtered_masks = []
        
        # Calculate image area if shape provided
        image_area = None
        if image_shape:
            image_area = image_shape[0] * image_shape[1]
        
        for mask in masks:
            # Check minimum area
            if mask.area < self.min_mask_area:
                continue
            
            # Check maximum area ratio if image area available
            if image_area and mask.area > (image_area * self.max_mask_area_ratio):
                continue
            
            # Check confidence threshold
            if mask.confidence < self.confidence_threshold:
                continue
            
            # Check stability score threshold
            if mask.stability_score < self.stability_score_threshold:
                continue
            
            filtered_masks.append(mask)
        
        return filtered_masks
    
    def select_primary_subject(self, masks: List[SegmentationMask]) -> Optional[SegmentationMask]:
        """
        Select the largest coherent mask as the primary manatee candidate.
        
        Implements requirement 2.2: "WHEN multiple objects are detected, 
        THE Segmentation_Engine SHALL select the largest mammalian shape as the primary manatee"
        
        Implements requirement 2.3: "WHEN no manatee is detected, 
        THE Segmentation_Engine SHALL return an empty mask with appropriate notification"
        
        Args:
            masks: List of filtered segmentation masks
            
        Returns:
            Primary mask representing the manatee, or None if no suitable mask found
        """
        if not masks:
            logging.info("No masks detected - returning None for empty result")
            return None
        
        # Sort masks by a combination of area and mammalian shape score
        scored_masks = []
        for mask in masks:
            mammalian_score = self._calculate_mammalian_shape_score(mask)
            # Combine area and shape score (weighted towards area for manatees)
            combined_score = (mask.area * 0.7) + (mammalian_score * mask.area * 0.3)
            scored_masks.append((mask, combined_score, mammalian_score))
        
        # Sort by combined score (highest first)
        scored_masks.sort(key=lambda x: x[1], reverse=True)
        
        # Try to find a mask that passes both coherence and mammalian shape validation
        for mask, score, mammalian_score in scored_masks:
            if (self._validate_mask_coherence(mask) and 
                self._validate_mammalian_characteristics(mask, mammalian_score)):
                logging.info(f"Selected primary subject with area {mask.area} and mammalian score {mammalian_score:.3f}")
                return mask
        
        # If no mask passes full validation, try just coherence validation
        for mask, score, mammalian_score in scored_masks:
            if self._validate_mask_coherence(mask):
                logging.warning(f"Selected mask with area {mask.area} but low mammalian score {mammalian_score:.3f}")
                return mask
        
        # If no mask passes coherence validation, return the highest scoring one
        # This ensures we always return something if masks are available
        best_mask = scored_masks[0][0]
        logging.warning(f"Selected mask with area {best_mask.area} despite failing validation checks")
        return best_mask
    
    def _validate_mask_coherence(self, mask: SegmentationMask) -> bool:
        """
        Validate that a mask represents a coherent shape suitable for a manatee.
        
        Args:
            mask: Segmentation mask to validate
            
        Returns:
            True if mask appears coherent, False otherwise
        """
        try:
            # Basic coherence checks
            if mask.area <= 0:
                return False
            
            # Check aspect ratio of bounding box
            x, y, w, h = mask.bbox
            if w <= 0 or h <= 0:
                return False
            
            aspect_ratio = max(w, h) / min(w, h)
            # Manatees are typically not extremely elongated
            if aspect_ratio > 5.0:
                return False
            
            # Check mask density (filled area vs bounding box area)
            bbox_area = w * h
            if bbox_area > 0:
                density = mask.area / bbox_area
                # Mask should fill a reasonable portion of its bounding box
                if density < 0.3:  # Less than 30% filled
                    return False
            
            return True
            
        except Exception as e:
            logging.warning(f"Failed to validate mask coherence: {e}")
            return False
    
    def _calculate_mammalian_shape_score(self, mask: SegmentationMask) -> float:
        """
        Calculate a score indicating how well a mask matches mammalian characteristics.
        
        Args:
            mask: Segmentation mask to score
            
        Returns:
            Score between 0.0 and 1.0, higher values indicate better mammalian characteristics
        """
        try:
            score = 0.0
            
            # Aspect ratio scoring (manatees are typically oval/elongated)
            x, y, w, h = mask.bbox
            if w > 0 and h > 0:
                aspect_ratio = max(w, h) / min(w, h)
                # Optimal aspect ratio for manatees is around 1.5-3.0
                if 1.5 <= aspect_ratio <= 3.0:
                    score += 0.3
                elif 1.0 <= aspect_ratio <= 4.0:
                    score += 0.2
                else:
                    score += 0.1
            
            # Size scoring (manatees should be reasonably large objects)
            if mask.area > 5000:  # Large object
                score += 0.3
            elif mask.area > 2000:  # Medium object
                score += 0.2
            else:  # Small object
                score += 0.1
            
            # Density scoring (solid shapes preferred)
            bbox_area = w * h
            if bbox_area > 0:
                density = mask.area / bbox_area
                if density > 0.6:  # Very solid
                    score += 0.2
                elif density > 0.4:  # Moderately solid
                    score += 0.15
                else:  # Sparse
                    score += 0.05
            
            # Confidence and stability scoring
            confidence_score = mask.confidence * 0.1
            stability_score = mask.stability_score * 0.1
            score += confidence_score + stability_score
            
            return min(score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logging.warning(f"Failed to calculate mammalian shape score: {e}")
            return 0.0
    
    def _validate_mammalian_characteristics(self, mask: SegmentationMask, mammalian_score: float) -> bool:
        """
        Validate that a mask has sufficient mammalian characteristics.
        
        Args:
            mask: Segmentation mask to validate
            mammalian_score: Pre-calculated mammalian shape score
            
        Returns:
            True if mask has good mammalian characteristics, False otherwise
        """
        # Require a minimum mammalian score
        min_mammalian_score = 0.4
        return mammalian_score >= min_mammalian_score
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded SAM model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_type": self.model_type,
            "checkpoint_path": self.checkpoint_path,
            "device": self.device,
            "model_loaded": self.sam_model is not None,
            "mask_generator_ready": self.mask_generator is not None,
            "integrated_filtering_enabled": self.enable_integrated_filtering,
            "mask_processor_ready": self.mask_processor is not None,
            "min_mask_area": self.min_mask_area,
            "max_mask_area_ratio": self.max_mask_area_ratio,
            "stability_score_threshold": self.stability_score_threshold,
            "confidence_threshold": self.confidence_threshold
        }
    
    def process_mask_with_filtering(self, mask: SegmentationMask, image: np.ndarray) -> SegmentationMask:
        """
        Process a segmentation mask with integrated waterline and reflection filtering.
        
        This method integrates reflection filtering with mask processing as required by task 6.3.
        
        Args:
            mask: Input segmentation mask
            image: Original RGB image
            
        Returns:
            Processed mask with waterline and reflection filtering applied
        """
        if not self.enable_integrated_filtering or self.mask_processor is None:
            logging.warning("Integrated filtering not enabled - returning original mask")
            return mask
        
        if mask is None or image is None:
            return mask
        
        try:
            # Use the mask processor to apply integrated filtering
            processed_mask = self.mask_processor.process_mask(mask, image)
            logging.info("Applied integrated waterline and reflection filtering to mask")
            return processed_mask
            
        except Exception as e:
            logging.error(f"Error processing mask with integrated filtering: {e}")
            return mask  # Return original mask if processing fails