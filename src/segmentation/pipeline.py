"""
Main segmentation pipeline for manatee body extraction.
"""

from pathlib import Path
from typing import Optional
import time
import numpy as np
import logging

from src.models.data_models import ProcessingResult, SegmentationMask
from src.utils.image_loader import ImageLoader
from src.utils.config_loader import ConfigLoader
from src.segmentation.sam_engine import SAMSegmentationEngine
from src.detection.waterline_detector import WaterlineDetector
from src.detection.reflection_filter import ReflectionFilter
from src.segmentation.mask_generator import MaskGenerator
from src.segmentation.quality_validator import QualityValidator


class SegmentationPipeline:
    """
    Main pipeline for processing manatee images and generating segmentation masks.
    
    This class orchestrates the entire segmentation workflow:
    1. Image loading and validation
    2. SAM-based segmentation
    3. Waterline detection and filtering
    4. Reflection removal
    5. Mask generation and output
    """
    
    def __init__(self, config_path: Path, model_type: str = "vit_h", 
                 checkpoint_path: Optional[Path] = None):
        """
        Initialize the segmentation pipeline.
        
        Args:
            config_path: Path to YAML configuration file
            model_type: SAM model type (vit_h, vit_l, vit_b)
            checkpoint_path: Optional path to model checkpoint
        """
        self.config_path = config_path
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        
        # Load configuration
        self.config_loader = ConfigLoader()
        try:
            self.config = self.config_loader.load_config(config_path)
            self.config = self.config_loader.merge_with_defaults(self.config)
        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {e}")
            logging.info("Using default configuration")
            self.config = self.config_loader.get_default_config()
        
        # Override config with command line parameters
        if model_type != "vit_h":
            self.config['sam']['model_type'] = model_type
        if checkpoint_path:
            self.config['sam']['checkpoint_path'] = str(checkpoint_path)
        
        # Initialize components with configuration
        self.image_loader = ImageLoader()
        self.sam_engine = SAMSegmentationEngine(
            self.config['sam']['model_type'], 
            self.config['sam']['checkpoint_path']
        )
        self.waterline_detector = WaterlineDetector()
        self.reflection_filter = ReflectionFilter()
        self.mask_generator = MaskGenerator()
        self.quality_validator = QualityValidator()
    
    def process_image(self, input_path: Path, output_path: Path) -> ProcessingResult:
        """
        Process a single image and generate segmentation mask.
        
        Args:
            input_path: Path to input PNG image
            output_path: Path where output mask will be saved
            
        Returns:
            ProcessingResult with success status and metadata
        """
        start_time = time.time()
        logger = logging.getLogger(__name__)
        
        try:
            logger.info(f"Starting image processing pipeline for: {input_path}")
            
            # Step 1: Load and validate image
            logger.info("Step 1: Loading and validating image")
            image = self.image_loader.load_image(str(input_path))
            if image is None:
                return ProcessingResult(
                    success=False,
                    mask=None,
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    error_message=f"Failed to load image: {input_path}"
                )
            
            image_info = self.image_loader.get_image_info(image)
            logger.info(f"Loaded image: {image_info.width}x{image_info.height}, {image_info.channels} channels")
            
            # Step 2: Generate initial segmentation masks using SAM
            logger.info("Step 2: Generating segmentation masks with SAM")
            masks = self.sam_engine.generate_masks(image)
            if not masks:
                return ProcessingResult(
                    success=False,
                    mask=None,
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    error_message="No objects detected in image"
                )
            
            logger.info(f"Generated {len(masks)} initial masks")
            
            # Step 3: Filter masks by size and select primary subject
            logger.info("Step 3: Filtering masks and selecting primary subject")
            filtered_masks = self.sam_engine.filter_by_size(masks, image.shape[:2])
            primary_mask = self.sam_engine.select_primary_subject(filtered_masks)
            
            if primary_mask is None:
                return ProcessingResult(
                    success=False,
                    mask=None,
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    error_message="No suitable manatee candidate detected"
                )
            
            logger.info(f"Selected primary mask with area {primary_mask.area} and confidence {primary_mask.confidence:.3f}")
            
            # Step 4: Detect waterline boundary
            logger.info("Step 4: Detecting waterline boundary")
            waterline = self.waterline_detector.detect_waterline(image)
            
            if waterline is not None:
                logger.info(f"Detected waterline with confidence {waterline.confidence:.3f}")
            else:
                logger.warning("No waterline detected - proceeding without waterline filtering")
            
            # Step 5: Generate final mask using MaskGenerator with integrated filtering
            logger.info("Step 5: Generating final binary mask with filtering")
            result = self.mask_generator.generate_mask_with_metadata(
                segmentation=primary_mask,
                image_shape=image.shape[:2],
                waterline=waterline,
                output_path=str(output_path),
                original_image=image
            )
            
            if not result.success:
                return result
            
            # Step 6: Apply additional reflection filtering if waterline was detected
            if waterline is not None:
                logger.info("Step 6: Applying reflection filtering")
                reflections = self.reflection_filter.detect_reflections(image, waterline)
                if reflections:
                    result.mask = self.reflection_filter.filter_mask_reflections(result.mask, reflections)
                    logger.info(f"Filtered {len(reflections)} reflection regions")
                    
                    # Adjust confidence based on reflection filtering
                    reflection_penalty = min(0.2, len(reflections) * 0.05)
                    result.confidence *= (1.0 - reflection_penalty)
                    result.confidence = max(0.0, min(1.0, result.confidence))
            
            # Step 7: Final quality validation and confidence calculation
            logger.info("Step 7: Final quality validation")
            coherence_score = self.quality_validator.validate_coherence(result.mask)
            final_confidence = self.quality_validator.calculate_confidence_score(
                primary_mask, result.mask, image
            )
            
            # Check if manual review is needed
            quality_flags = self.quality_validator.flag_for_manual_review(
                final_confidence, coherence_score, result.mask
            )
            
            if quality_flags["needs_review"]:
                logger.warning(f"Segmentation flagged for manual review: {quality_flags['reasons']}")
            else:
                logger.info(f"Segmentation passed quality validation")
            
            # Update result with final values
            result.confidence = final_confidence
            result.processing_time = time.time() - start_time
            
            # Log final statistics
            mask_stats = self.mask_generator.get_mask_statistics(result.mask)
            logger.info(f"Final mask statistics: {mask_stats['manatee_pixels']} manatee pixels "
                       f"({mask_stats['manatee_percentage']:.1f}% of image)")
            logger.info(f"Processing completed successfully in {result.processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Pipeline processing error: {e}", exc_info=True)
            return ProcessingResult(
                success=False,
                mask=None,
                confidence=0.0,
                processing_time=processing_time,
                error_message=f"Pipeline error: {str(e)}"
            )
    
    def apply_reflection_filtering(self, mask: np.ndarray, image: np.ndarray, waterline) -> np.ndarray:
        """
        Apply reflection filtering to a segmentation mask.
        
        This method integrates reflection filtering with mask processing as required by task 6.3.
        It detects reflections in the image and removes them from the segmentation mask.
        
        Args:
            mask: Input binary segmentation mask
            image: Original RGB image
            waterline: Detected waterline boundary (can be None)
            
        Returns:
            Filtered mask with reflections removed
        """
        if mask is None or image is None:
            return mask
        
        if waterline is None:
            logging.warning("No waterline provided for reflection filtering - returning original mask")
            return mask
        
        try:
            # Detect reflections using the reflection filter
            reflections = self.reflection_filter.detect_reflections(image, waterline)
            
            if not reflections:
                logging.info("No reflections detected")
                return mask
            
            # Apply reflection filtering to the mask
            filtered_mask = self.reflection_filter.filter_mask_reflections(mask, reflections)
            
            logging.info(f"Applied reflection filtering - removed {len(reflections)} reflection regions")
            return filtered_mask
            
        except Exception as e:
            logging.error(f"Error during reflection filtering: {e}")
            return mask  # Return original mask if filtering fails
    
    def get_pipeline_info(self) -> dict:
        """
        Get information about the pipeline configuration and component status.
        
        Returns:
            Dictionary with pipeline information
        """
        return {
            "config_path": str(self.config_path),
            "model_type": self.model_type,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "components": {
                "image_loader": self.image_loader is not None,
                "sam_engine": self.sam_engine is not None,
                "waterline_detector": self.waterline_detector is not None,
                "reflection_filter": self.reflection_filter is not None,
                "mask_generator": self.mask_generator is not None,
                "quality_validator": self.quality_validator is not None
            },
            "sam_info": self.sam_engine.get_model_info() if self.sam_engine else None,
            "pipeline_version": "1.0.0",
            "supported_formats": ["PNG"],
            "processing_stages": [
                "Image Loading & Validation",
                "SAM Segmentation",
                "Mask Filtering & Selection",
                "Waterline Detection",
                "Binary Mask Generation",
                "Reflection Filtering",
                "Quality Validation"
            ]
        }