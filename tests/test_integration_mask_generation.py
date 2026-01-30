"""
Integration test for mask generation workflow.
"""

import numpy as np
import tempfile
import os
from src.segmentation import MaskGenerator, QualityValidator
from src.models.data_models import SegmentationMask, WaterlineBoundary


def test_complete_mask_generation_workflow():
    """Test the complete mask generation workflow."""
    
    # Create test data
    image_shape = (200, 300)  # height, width
    
    # Create a realistic segmentation mask (oval shape for manatee)
    mask = np.zeros(image_shape, dtype=bool)
    center_y, center_x = 100, 150
    for y in range(image_shape[0]):
        for x in range(image_shape[1]):
            # Create an elliptical shape
            if ((y - center_y) / 40) ** 2 + ((x - center_x) / 60) ** 2 <= 1:
                mask[y, x] = True
    
    segmentation = SegmentationMask(
        mask=mask,
        bbox=(90, 60, 120, 80),  # x, y, width, height
        area=int(np.sum(mask)),
        confidence=0.85,
        stability_score=0.92
    )
    
    # Create a waterline (horizontal line at y=150)
    waterline = WaterlineBoundary(
        points=[(0, 150), (300, 150)],
        equation=(0.0, 1.0, -150.0),  # y = 150
        confidence=0.88,
        water_region_below=True
    )
    
    # Create synthetic original image for quality validation
    original_image = np.random.randint(0, 255, (*image_shape, 3), dtype=np.uint8)
    
    # Initialize mask generator
    mask_generator = MaskGenerator()
    
    # Test complete workflow with file output
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        output_path = tmp_file.name
    
    try:
        # Generate mask with all features
        result = mask_generator.generate_mask_with_metadata(
            segmentation=segmentation,
            image_shape=image_shape,
            waterline=waterline,
            output_path=output_path,
            original_image=original_image
        )
        
        # Verify result
        assert result.success is True
        assert result.mask is not None
        assert result.mask.shape == image_shape
        assert 0.0 <= result.confidence <= 1.0
        assert result.processing_time >= 0.0
        assert result.error_message is None
        
        # Verify file was created
        assert os.path.exists(output_path)
        
        # Generate quality report
        quality_report = mask_generator.generate_quality_report(
            segmentation, result.mask, original_image
        )
        
        # Verify quality report structure
        required_keys = [
            "overall_quality", "confidence_score", "coherence_score",
            "mask_statistics", "quality_flags", "recommendations"
        ]
        
        for key in required_keys:
            assert key in quality_report
        
        print(f"✓ Mask generation successful")
        print(f"✓ Output file created: {output_path}")
        print(f"✓ Final confidence: {result.confidence:.3f}")
        print(f"✓ Overall quality: {quality_report['overall_quality']}")
        print(f"✓ Coherence score: {quality_report['coherence_score']:.3f}")
        print(f"✓ Processing time: {result.processing_time:.3f}s")
        
        if quality_report['quality_flags']['needs_review']:
            print(f"⚠ Manual review recommended: {quality_report['quality_flags']['reasons']}")
        else:
            print("✓ Quality validation passed")
        
        return True
        
    finally:
        # Clean up
        if os.path.exists(output_path):
            os.unlink(output_path)


if __name__ == "__main__":
    test_complete_mask_generation_workflow()
    print("Integration test completed successfully!")