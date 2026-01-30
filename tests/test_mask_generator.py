"""
Tests for the MaskGenerator class.
"""

import numpy as np
import pytest
import tempfile
import os
from PIL import Image

from src.segmentation.mask_generator import MaskGenerator
from src.segmentation.quality_validator import QualityValidator
from src.models.data_models import SegmentationMask, WaterlineBoundary


class TestMaskGenerator:
    """Test cases for MaskGenerator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mask_generator = MaskGenerator()
        
        # Create a simple test segmentation mask
        test_mask = np.zeros((100, 100), dtype=bool)
        test_mask[30:70, 40:80] = True  # Rectangle in the middle
        
        self.test_segmentation = SegmentationMask(
            mask=test_mask,
            bbox=(40, 30, 40, 40),
            area=1600,
            confidence=0.8,
            stability_score=0.9
        )
        
        self.test_image_shape = (100, 100)
    
    def test_create_binary_mask_basic(self):
        """Test basic binary mask creation."""
        binary_mask = self.mask_generator.create_binary_mask(
            self.test_segmentation, self.test_image_shape
        )
        
        assert binary_mask.shape == self.test_image_shape
        assert binary_mask.dtype == np.uint8
        assert np.sum(binary_mask == 255) == 1600  # Area of the test mask
        assert np.sum(binary_mask == 0) == 10000 - 1600  # Background pixels
    
    def test_create_binary_mask_invalid_input(self):
        """Test binary mask creation with invalid inputs."""
        with pytest.raises(ValueError):
            self.mask_generator.create_binary_mask(None, self.test_image_shape)
        
        with pytest.raises(ValueError):
            self.mask_generator.create_binary_mask(self.test_segmentation, (0, 100))
    
    def test_save_mask_success(self):
        """Test successful mask saving."""
        binary_mask = self.mask_generator.create_binary_mask(
            self.test_segmentation, self.test_image_shape
        )
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            success = self.mask_generator.save_mask(binary_mask, tmp_path)
            assert success is True
            assert os.path.exists(tmp_path)
            
            # Verify the saved image
            saved_image = Image.open(tmp_path)
            saved_array = np.array(saved_image)
            assert saved_array.shape == self.test_image_shape
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_save_mask_invalid_input(self):
        """Test mask saving with invalid inputs."""
        with pytest.raises(ValueError):
            self.mask_generator.save_mask(None, "test.png")
        
        with pytest.raises(ValueError):
            self.mask_generator.save_mask(np.zeros((10, 10)), "")
    
    def test_apply_waterline_filter(self):
        """Test waterline filtering functionality."""
        binary_mask = np.ones((100, 100), dtype=np.uint8) * 255
        
        # Create a horizontal waterline at y=50
        waterline = WaterlineBoundary(
            points=[(0, 50), (100, 50)],
            equation=(0.0, 1.0, -50.0),  # y = 50
            confidence=0.9,
            water_region_below=True
        )
        
        filtered_mask = self.mask_generator.apply_waterline_filter(binary_mask, waterline)
        
        # Check that pixels below y=50 are removed
        assert np.all(filtered_mask[51:, :] == 0)  # Below waterline should be black
        assert np.any(filtered_mask[:50, :] == 255)  # Above waterline should have white pixels
    
    def test_generate_mask_with_metadata(self):
        """Test complete mask generation with metadata."""
        result = self.mask_generator.generate_mask_with_metadata(
            self.test_segmentation, self.test_image_shape
        )
        
        assert result.success is True
        assert result.mask is not None
        assert result.mask.shape == self.test_image_shape
        assert 0.0 <= result.confidence <= 1.0
        assert result.processing_time >= 0.0
        assert result.error_message is None
    
    def test_validate_output_dimensions(self):
        """Test output dimension validation."""
        mask = np.zeros((100, 100))
        assert self.mask_generator.validate_output_dimensions(mask, (100, 100)) is True
        assert self.mask_generator.validate_output_dimensions(mask, (50, 50)) is False
        assert self.mask_generator.validate_output_dimensions(None, (100, 100)) is False
    
    def test_get_mask_statistics(self):
        """Test mask statistics calculation."""
        binary_mask = np.zeros((100, 100), dtype=np.uint8)
        binary_mask[30:70, 40:80] = 255
        
        stats = self.mask_generator.get_mask_statistics(binary_mask)
        
        assert stats["shape"] == (100, 100)
        assert stats["total_pixels"] == 10000
        assert stats["manatee_pixels"] == 1600
        assert stats["background_pixels"] == 8400
        assert abs(stats["manatee_percentage"] - 16.0) < 0.1
        assert stats["min_value"] == 0
        assert stats["max_value"] == 255


class TestQualityValidator:
    """Test cases for QualityValidator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = QualityValidator()
        
        # Create test masks
        self.good_mask = np.zeros((100, 100), dtype=bool)
        self.good_mask[30:70, 40:80] = True  # Nice rectangular shape
        
        self.fragmented_mask = np.zeros((100, 100), dtype=bool)
        self.fragmented_mask[10:15, 10:15] = True  # Small fragment 1
        self.fragmented_mask[80:85, 80:85] = True  # Small fragment 2
        
        self.test_segmentation = SegmentationMask(
            mask=self.good_mask,
            bbox=(40, 30, 40, 40),
            area=1600,
            confidence=0.8,
            stability_score=0.9
        )
    
    def test_validate_coherence_good_mask(self):
        """Test coherence validation with a good mask."""
        coherence = self.validator.validate_coherence(self.good_mask)
        assert 0.0 <= coherence <= 1.0
        assert coherence > 0.5  # Should be reasonably coherent
    
    def test_validate_coherence_fragmented_mask(self):
        """Test coherence validation with a fragmented mask."""
        coherence = self.validator.validate_coherence(self.fragmented_mask)
        assert 0.0 <= coherence <= 1.0
        # Fragmented mask should have lower coherence, but exact value depends on implementation
    
    def test_validate_coherence_empty_mask(self):
        """Test coherence validation with empty mask."""
        empty_mask = np.zeros((100, 100), dtype=bool)
        coherence = self.validator.validate_coherence(empty_mask)
        assert coherence == 0.0
    
    def test_calculate_confidence_score(self):
        """Test confidence score calculation."""
        confidence = self.validator.calculate_confidence_score(
            self.test_segmentation, self.good_mask
        )
        assert 0.0 <= confidence <= 1.0
    
    def test_flag_for_manual_review(self):
        """Test manual review flagging."""
        flags = self.validator.flag_for_manual_review(0.8, 0.8, self.good_mask)
        assert "needs_review" in flags
        assert "reasons" in flags
        assert "severity" in flags
        assert isinstance(flags["needs_review"], bool)
        
        # Test low confidence flagging
        low_flags = self.validator.flag_for_manual_review(0.3, 0.3, self.good_mask)
        assert low_flags["needs_review"] is True
        assert low_flags["severity"] in ["low", "medium", "high"]
    
    def test_generate_quality_report(self):
        """Test quality report generation."""
        report = self.validator.generate_quality_report(
            self.test_segmentation, self.good_mask
        )
        
        required_keys = [
            "overall_quality", "confidence_score", "coherence_score",
            "mask_statistics", "quality_flags", "recommendations"
        ]
        
        for key in required_keys:
            assert key in report
        
        assert report["overall_quality"] in ["excellent", "good", "fair", "poor", "error"]
        assert 0.0 <= report["confidence_score"] <= 1.0
        assert 0.0 <= report["coherence_score"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])