#!/usr/bin/env python3
"""
Demo script to showcase the complete manatee segmentation pipeline.

This script demonstrates the end-to-end functionality of the integrated
segmentation system without requiring the full SAM model to be loaded.
"""

import sys
from pathlib import Path
import logging

# Add src to path for imports
sys.path.append('.')

from src.segmentation.pipeline import SegmentationPipeline
from src.utils.logger import setup_logger


def main():
    """Demonstrate the pipeline integration."""
    
    # Setup logging
    logger = setup_logger(verbose=True)
    
    logger.info("=" * 60)
    logger.info("MANATEE SEGMENTATION PIPELINE DEMO")
    logger.info("=" * 60)
    
    try:
        # Initialize pipeline
        logger.info("1. Initializing segmentation pipeline...")
        config_path = Path('config/default.yaml')
        
        pipeline = SegmentationPipeline(
            config_path=config_path,
            model_type='vit_h'
        )
        
        logger.info("✓ Pipeline initialized successfully")
        
        # Show pipeline information
        logger.info("\n2. Pipeline Configuration:")
        info = pipeline.get_pipeline_info()
        
        logger.info(f"   Version: {info['pipeline_version']}")
        logger.info(f"   Model Type: {info['model_type']}")
        logger.info(f"   Config File: {info['config_path']}")
        logger.info(f"   Checkpoint: {info['checkpoint_path'] or 'Auto-download'}")
        
        logger.info("\n   Components Status:")
        for component, status in info['components'].items():
            status_str = "✓" if status else "✗"
            logger.info(f"     {status_str} {component.replace('_', ' ').title()}")
        
        logger.info(f"\n   Processing Stages ({len(info['processing_stages'])}):")
        for i, stage in enumerate(info['processing_stages'], 1):
            logger.info(f"     {i}. {stage}")
        
        # Test configuration loading
        logger.info("\n3. Configuration Validation:")
        config_sections = ['sam', 'segmentation', 'waterline', 'reflection_filter', 'output', 'quality']
        for section in config_sections:
            if section in pipeline.config:
                logger.info(f"   ✓ {section.replace('_', ' ').title()} configuration loaded")
            else:
                logger.info(f"   ✗ {section.replace('_', ' ').title()} configuration missing")
        
        # Test component integration
        logger.info("\n4. Component Integration Test:")
        
        # Test image loader
        if hasattr(pipeline.image_loader, 'load_image'):
            logger.info("   ✓ Image Loader - load_image method available")
        
        # Test SAM engine
        if hasattr(pipeline.sam_engine, 'generate_masks'):
            logger.info("   ✓ SAM Engine - generate_masks method available")
        
        # Test waterline detector
        if hasattr(pipeline.waterline_detector, 'detect_waterline'):
            logger.info("   ✓ Waterline Detector - detect_waterline method available")
        
        # Test reflection filter
        if hasattr(pipeline.reflection_filter, 'detect_reflections'):
            logger.info("   ✓ Reflection Filter - detect_reflections method available")
        
        # Test mask generator
        if hasattr(pipeline.mask_generator, 'create_binary_mask'):
            logger.info("   ✓ Mask Generator - create_binary_mask method available")
        
        # Test quality validator
        if hasattr(pipeline.quality_validator, 'validate_coherence'):
            logger.info("   ✓ Quality Validator - validate_coherence method available")
        
        # Test error handling
        logger.info("\n5. Error Handling Test:")
        result = pipeline.process_image(
            Path('non_existent_image.png'),
            Path('test_output.png')
        )
        
        if not result.success:
            logger.info("   ✓ Error handling works correctly")
            logger.info(f"   ✓ Error message: {result.error_message}")
            logger.info(f"   ✓ Processing time recorded: {result.processing_time:.4f}s")
        else:
            logger.warning("   ⚠ Expected error handling test to fail")
        
        # Test with real image if available
        logger.info("\n6. Real Image Test:")
        test_image = Path('manatee.PNG')
        if test_image.exists():
            logger.info(f"   Found test image: {test_image}")
            logger.info("   Note: Full processing requires SAM model weights")
            logger.info("   Use 'python main.py manatee.PNG output.png --validate-only' to test image loading")
        else:
            logger.info("   No test image found (manatee.PNG)")
            logger.info("   Pipeline is ready for processing when image is provided")
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE INTEGRATION DEMO COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        logger.info("\nNext Steps:")
        logger.info("1. Use 'python main.py --info' to see pipeline information")
        logger.info("2. Use 'python main.py image.png output.png' to process an image")
        logger.info("3. Use 'python main.py image.png output.png --validate-only' to validate input")
        logger.info("4. Use 'python main.py --help' for all available options")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        logger.exception("Full traceback:")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())