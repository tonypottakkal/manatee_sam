"""
Configuration loading utilities for the manatee segmentation system.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """
    Loads and validates configuration files for the segmentation system.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_config(self, config_path: Path) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Dictionary containing configuration parameters
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
            ValueError: If required configuration sections are missing
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                config = {}
            
            # Validate required sections
            self._validate_config(config)
            
            self.logger.info(f"Loaded configuration from: {config_path}")
            return config
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in config file {config_path}: {e}")
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate that required configuration sections are present.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If required sections are missing
        """
        required_sections = ['sam', 'segmentation', 'waterline', 'reflection_filter', 'output', 'quality']
        
        for section in required_sections:
            if section not in config:
                self.logger.warning(f"Missing configuration section: {section}")
        
        # Validate SAM configuration
        if 'sam' in config:
            sam_config = config['sam']
            if 'model_type' not in sam_config:
                self.logger.warning("Missing SAM model_type in configuration")
        
        # Validate segmentation parameters
        if 'segmentation' in config:
            seg_config = config['segmentation']
            required_seg_params = ['min_mask_area', 'max_masks', 'stability_score_thresh']
            for param in required_seg_params:
                if param not in seg_config:
                    self.logger.warning(f"Missing segmentation parameter: {param}")
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration values.
        
        Returns:
            Dictionary with default configuration
        """
        return {
            'sam': {
                'model_type': 'vit_h',
                'checkpoint_path': None,
                'device': 'auto'
            },
            'segmentation': {
                'min_mask_area': 1000,
                'max_masks': 50,
                'stability_score_thresh': 0.95,
                'box_nms_thresh': 0.7
            },
            'waterline': {
                'enabled': True,
                'hsv_water_ranges': {
                    'hue': [100, 130],
                    'saturation': [50, 255],
                    'value': [50, 255]
                },
                'edge_detection': {
                    'low_threshold': 50,
                    'high_threshold': 150
                },
                'hough_lines': {
                    'rho': 1,
                    'theta_resolution': 180,
                    'threshold': 100,
                    'min_line_length': 100,
                    'max_line_gap': 10
                }
            },
            'reflection_filter': {
                'enabled': True,
                'symmetry_threshold': 0.7,
                'color_similarity_threshold': 0.8,
                'morphology': {
                    'kernel_size': 5,
                    'iterations': 2
                }
            },
            'output': {
                'format': 'png',
                'compression_level': 6,
                'preserve_dimensions': True
            },
            'quality': {
                'min_confidence': 0.5,
                'coherence_threshold': 0.8,
                'flag_for_review_threshold': 0.3
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'save_to_file': False,
                'log_file': 'logs/segmentation.log'
            }
        }
    
    def merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge loaded configuration with default values.
        
        Args:
            config: Loaded configuration dictionary
            
        Returns:
            Merged configuration with defaults filled in
        """
        default_config = self.get_default_config()
        
        def deep_merge(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively merge dictionaries."""
            result = default.copy()
            
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            
            return result
        
        return deep_merge(default_config, config)
    
    def save_config(self, config: Dict[str, Any], output_path: Path) -> bool:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary to save
            output_path: Path where to save the configuration
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Saved configuration to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False