# Manatee Segmentation System

An automated manatee body segmentation system using Meta's [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything). This system identifies and extracts only the visible manatee body above the waterline, excluding water areas and reflections to produce clean segmentation masks.

## Features

- **Zero-shot segmentation** using SAM (Segment Anything Model)
- **Waterline detection** to exclude underwater portions
- **Reflection filtering** to remove water reflections
- **Quality validation** with confidence scoring
- **Command-line interface** for batch processing
- **Configurable parameters** via YAML configuration

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd manatee-segmentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Download SAM model checkpoints** (required):
   
   Download the appropriate SAM model checkpoint from the [official SAM repository](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints):
   
   - **ViT-H SAM model** (recommended, ~2.6GB): `sam_vit_h_4b8939.pth`
   - **ViT-L SAM model** (~1.2GB): `sam_vit_l_0b3195.pth`  
   - **ViT-B SAM model** (~375MB): `sam_vit_b_01ec64.pth`
   
   Place the downloaded checkpoint in the `sam_checkpoints/` directory:
   ```bash
   mkdir -p sam_checkpoints
   # Download your chosen model checkpoint to this directory
   ```
   
   **Note**: The system will auto-download checkpoints if not found locally, but manual download is recommended for better control and faster startup.

4. Install the package:
```bash
pip install -e .
```

## Usage

### Command Line Interface

Process a single image:
```bash
python main.py input_image.png output_mask.png
```

With custom configuration:
```bash
python main.py input_image.png output_mask.png --config config/custom.yaml
```

Specify SAM model type and checkpoint:
```bash
python main.py input_image.png output_mask.png --model-type vit_h --checkpoint sam_checkpoints/sam_vit_h_4b8939.pth
```

### Python API

```python
from src.segmentation.pipeline import SegmentationPipeline

# Initialize pipeline
pipeline = SegmentationPipeline(
    config_path="config/default.yaml",
    model_type="vit_h"
)

# Process image
result = pipeline.process_image("input.png", "output_mask.png")

if result.success:
    print(f"Segmentation completed with confidence: {result.confidence}")
else:
    print(f"Segmentation failed: {result.error_message}")
```

## Configuration

The system uses YAML configuration files to control processing parameters. See `config/default.yaml` for all available options.

Key configuration sections:
- **SAM Model**: Model type, checkpoint path, device settings
- **Segmentation**: Mask filtering and selection parameters
- **Waterline Detection**: HSV ranges, edge detection, Hough transforms
- **Reflection Filtering**: Symmetry and color similarity thresholds
- **Quality Validation**: Confidence and coherence thresholds

## Project Structure

```
manatee-segmentation/
├── src/
│   ├── models/          # Data models and structures
│   ├── segmentation/    # SAM segmentation engine
│   ├── detection/       # Waterline and reflection detection
│   └── utils/           # Utility functions
├── tests/               # Test suite
├── config/              # Configuration files
├── main.py              # Command-line entry point
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- [Segment Anything Model](https://github.com/facebookresearch/segment-anything) - see installation section for checkpoint download
- See `requirements.txt` for complete list

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Citation

If you use this system in your research, please cite both this work and the original SAM paper:

```bibtex
@software{manatee_segmentation,
  title={Manatee Segmentation System},
  author={Tony Pottakkal},
  year={2026},
  url={https://github.com/tonypottakkal/manatee_sam},
  note={Developed with AI-assisted coding using Kiro}
}

@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```