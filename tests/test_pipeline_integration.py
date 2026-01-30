"""
Integration tests for the complete manatee segmentation pipeline.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

from src.segmentation.pipeline import SegmentationPipeline
from src.models.data_models import ProcessingResult


class TestPipelineIntegration:
    """Test the complete segmentation pipeline integration."""
    
    def test_pipeline_initialization(self):
        """Test that the pipeline initializes all components correctly."""
        config_path = Path('config/default.yaml')
        
        pipeline = SegmentationPipeline(
            config_path=config_path,
            model_type='vit_h'
        )
        
        # Check that all components are initialized
        assert pipeline.image_loader is not None
        assert pipeline.sam_engine is not None
        assert pipeline.waterline_detector is not None
        assert pipeline.reflection_filter is not None
        assert pipeline.mask_generator is not None
        assert pipeline.quality_validator is not None
        
        # Check pipeline info
        info = pipeline.get_pipeline_info()
        assert info['pipeline_version'] == '1.0.0'
        assert info['model_type'] == 'vit_h'
        assert all(info['components'].values())  # All components should be True
        assert len(info['processing_stages']) == 7
    
    def test_pipeline_with_missing_config(self):
        """Test pipeline initialization with missing config file."""
        non_existent_config = Path('non_existent_config.yaml')
        
        # Should not raise an exception, should use defaults
        pipeline = SegmentationPipeline(
            config_path=non_existent_config,
            model_type='vit_l'
        )
        
        assert pipeline.config is not None
        assert pipeline.config['sam']['model_type'] == 'vit_l'  # Should use command line override
    
    def test_pipeline_config_override(self):
        """Test that command line parameters override config file."""
        config_path = Path('config/default.yaml')
        
        pipeline = SegmentationPipeline(
            config_path=config_path,
            model_type='vit_l',  # Override default vit_h
            checkpoint_path=Path('custom_checkpoint.pth')
        )
        
        assert pipeline.config['sam']['model_type'] == 'vit_l'
        assert pipeline.config['sam']['checkpoint_path'] == 'custom_checkpoint.pth'
    
    def test_pipeline_error_handling_invalid_image(self):
        """Test pipeline error handling with invalid image path."""
        config_path = Path('config/default.yaml')
        pipeline = SegmentationPipeline(config_path=config_path)
        
        # Test with non-existent file
        result = pipeline.process_image(
            Path('non_existent_image.png'),
            Path('output.png')
        )
        
        assert not result.success
        assert result.mask is None
        assert result.confidence == 0.0
        assert 'Invalid image format or path' in result.error_message
        assert result.processing_time > 0
    
    def test_pipeline_component_integration(self):
        """Test that pipeline components are properly integrated."""
        config_path = Path('config/default.yaml')
        pipeline = SegmentationPipeline(config_path=config_path)
        
        # Test that components can be accessed and have expected methods
        assert hasattr(pipeline.image_loader, 'load_image')
        assert hasattr(pipeline.sam_engine, 'generate_masks')
        assert hasattr(pipeline.waterline_detector, 'detect_waterline')
        assert hasattr(pipeline.reflection_filter, 'detect_reflections')
        assert hasattr(pipeline.mask_generator, 'create_binary_mask')
        assert hasattr(pipeline.quality_validator, 'validate_coherence')
    
    def test_pipeline_configuration_loading(self):
        """Test that configuration is properly loaded and merged."""
        config_path = Path('config/default.yaml')
        pipeline = SegmentationPipeline(config_path=config_path)
        
        # Check that configuration sections exist
        assert 'sam' in pipeline.config
        assert 'segmentation' in pipeline.config
        assert 'waterline' in pipeline.config
        assert 'reflection_filter' in pipeline.config
        assert 'output' in pipeline.config
        assert 'quality' in pipeline.config
        
        # Check specific configuration values
        assert pipeline.config['sam']['model_type'] in ['vit_h', 'vit_l', 'vit_b']
        assert isinstance(pipeline.config['segmentation']['min_mask_area'], int)
        assert isinstance(pipeline.config['quality']['min_confidence'], float)
    
    @pytest.mark.skipif(not Path('manatee.PNG').exists(), reason="Test image not available")
    def test_pipeline_with_real_image(self):
        """Test pipeline with actual manatee image (if available)."""
        config_path = Path('config/default.yaml')
        pipeline = SegmentationPipeline(config_path=config_path)
        
        input_path = Path('manatee.PNG')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            output_path = Path(tmp_file.name)
        
        try:
            # This test may take a while due to SAM model loading
            # In a real test environment, you might want to mock the SAM engine
            result = pipeline.process_image(input_path, output_path)
            
            # The result might succeed or fail depending on model availability
            # But it should always return a valid ProcessingResult
            assert isinstance(result, ProcessingResult)
            assert isinstance(result.success, bool)
            assert isinstance(result.confidence, float)
            assert isinstance(result.processing_time, float)
            assert result.processing_time > 0
            
            if result.success:
                assert result.mask is not None
                assert isinstance(result.mask, np.ndarray)
                assert result.confidence >= 0.0
                assert result.confidence <= 1.0
                assert output_path.exists()
            else:
                assert result.error_message is not None
                assert isinstance(result.error_message, str)
                
        finally:
            # Clean up temporary file
            if output_path.exists():
                os.unlink(output_path)
    
    def test_pipeline_apply_reflection_filtering_method(self):
        """Test the apply_reflection_filtering method integration."""
        config_path = Path('config/default.yaml')
        pipeline = SegmentationPipeline(config_path=config_path)
        
        # Create dummy data
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test with no waterline
        result_mask = pipeline.apply_reflection_filtering(mask, image, None)
        assert np.array_equal(result_mask, mask)  # Should return original mask
        
        # Test with None inputs
        result_mask = pipeline.apply_reflection_filtering(None, image, None)
        assert result_mask is None
        
        result_mask = pipeline.apply_reflection_filtering(mask, None, None)
        assert np.array_equal(result_mask, mask)