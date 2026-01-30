#!/usr/bin/env python3
"""
Manatee Segmentation System - Main Entry Point

This script provides a command-line interface for processing manatee images
and generating segmentation masks using the Segment Anything Model (SAM).
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

import click
from src.utils.logger import setup_logger
from src.segmentation.pipeline import SegmentationPipeline


@click.command()
@click.argument('input_path', type=click.Path(path_type=Path), required=False)
@click.argument('output_path', type=click.Path(path_type=Path), required=False)
@click.option('--config', '-c', type=click.Path(exists=True, path_type=Path),
              default='config/default.yaml', help='Configuration file path')
@click.option('--model-type', '-m', default='vit_h',
              type=click.Choice(['vit_h', 'vit_l', 'vit_b']),
              help='SAM model type to use')
@click.option('--checkpoint', type=click.Path(exists=True, path_type=Path),
              help='Path to SAM model checkpoint')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--log-file', type=click.Path(path_type=Path),
              help='Save logs to specified file')
@click.option('--info', is_flag=True, help='Show pipeline information and exit')
@click.option('--validate-only', is_flag=True, help='Only validate input without processing')
def main(input_path: Optional[Path], output_path: Optional[Path], config: Path, model_type: str,
         checkpoint: Optional[Path], verbose: bool, log_file: Optional[Path],
         info: bool, validate_only: bool):
    """
    Process a manatee image and generate a segmentation mask.
    
    INPUT_PATH: Path to the input PNG image
    OUTPUT_PATH: Path where the output mask will be saved
    
    Examples:
        python main.py input.png output_mask.png
        python main.py input.png output_mask.png --verbose --model-type vit_l
        python main.py input.png output_mask.png --checkpoint sam_checkpoints/sam_vit_h_4b8939.pth
    """
    # Setup logging
    logger = setup_logger(verbose=verbose, log_file=log_file)
    
    try:
        # Initialize the segmentation pipeline
        logger.info("Initializing segmentation pipeline...")
        pipeline = SegmentationPipeline(
            config_path=config,
            model_type=model_type,
            checkpoint_path=checkpoint
        )
        
        # Show pipeline information if requested
        if info:
            pipeline_info = pipeline.get_pipeline_info()
            logger.info("Pipeline Information:")
            logger.info(f"  Version: {pipeline_info.get('pipeline_version', 'Unknown')}")
            logger.info(f"  Model Type: {pipeline_info['model_type']}")
            logger.info(f"  Config: {pipeline_info['config_path']}")
            logger.info(f"  Checkpoint: {pipeline_info['checkpoint_path'] or 'Auto-download'}")
            logger.info("  Components:")
            for component, status in pipeline_info['components'].items():
                status_str = "✓" if status else "✗"
                logger.info(f"    {status_str} {component}")
            logger.info("  Processing Stages:")
            for i, stage in enumerate(pipeline_info['processing_stages'], 1):
                logger.info(f"    {i}. {stage}")
            return
        
        # Validate required arguments for processing
        if not input_path or not output_path:
            logger.error("INPUT_PATH and OUTPUT_PATH are required for processing")
            logger.info("Use --info to show pipeline information without processing")
            sys.exit(1)
        
        # Check if input file exists
        if not input_path.exists():
            logger.error(f"Input file does not exist: {input_path}")
            sys.exit(1)
        
        # Validate input file format
        if not str(input_path).lower().endswith('.png'):
            logger.error("Input file must be a PNG image")
            sys.exit(1)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate input only if requested
        if validate_only:
            logger.info(f"Validating input file: {input_path}")
            # This would use the image loader to validate the file
            from src.utils.image_loader import ImageLoader
            loader = ImageLoader()
            image = loader.load_image(str(input_path))
            if image is not None:
                info = loader.get_image_info(image)
                logger.info(f"✓ Valid PNG image: {info.width}x{info.height}, {info.channels} channels")
                logger.info(f"  File size: {info.file_size} bytes")
            else:
                logger.error("✗ Invalid or corrupted image file")
                sys.exit(1)
            return
        
        logger.info(f"Processing image: {input_path}")
        logger.info(f"Output will be saved to: {output_path}")
        
        # Process the image
        result = pipeline.process_image(input_path, output_path)
        
        if result.success:
            logger.info("=" * 50)
            logger.info("SEGMENTATION COMPLETED SUCCESSFULLY!")
            logger.info("=" * 50)
            logger.info(f"Output saved to: {output_path}")
            logger.info(f"Confidence score: {result.confidence:.3f}")
            logger.info(f"Processing time: {result.processing_time:.2f} seconds")
            
            # Show mask statistics if available
            if result.mask is not None:
                from src.segmentation.mask_generator import MaskGenerator
                mask_gen = MaskGenerator()
                stats = mask_gen.get_mask_statistics(result.mask)
                logger.info(f"Mask statistics:")
                logger.info(f"  Total pixels: {stats['total_pixels']:,}")
                logger.info(f"  Manatee pixels: {stats['manatee_pixels']:,}")
                logger.info(f"  Coverage: {stats['manatee_percentage']:.1f}%")
            
            # Quality assessment
            if result.confidence < 0.3:
                logger.warning("⚠️  Low confidence result - manual review recommended")
            elif result.confidence < 0.6:
                logger.info("ℹ️  Moderate confidence result")
            else:
                logger.info("✓ High confidence result")
                
        else:
            logger.error("=" * 50)
            logger.error("SEGMENTATION FAILED")
            logger.error("=" * 50)
            logger.error(f"Error: {result.error_message}")
            logger.error(f"Processing time: {result.processing_time:.2f} seconds")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(130)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()