# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAM 2 (Segment Anything Model 2) is Meta AI's foundation model for promptable visual segmentation in images and videos. This repository contains the core model implementation, utilities, and various pipeline implementations for video processing.

## Installation and Setup

Install SAM 2 with demo dependencies:
```bash
pip install -e ".[demo]"
```

For development with linting tools:
```bash
pip install -e ".[dev]"
```

Download model checkpoints:
```bash
cd checkpoints && ./download_ckpts.sh && cd ..
```

## Development Commands

### Linting
Format code using the project's formatting tools:
```bash
ufmt format .
```

The project requires specific versions: `black==24.2.0`, `usort==1.0.2`, and `ufmt==2.0.0b2`

### Environment Variables
- `SAM2_BUILD_CUDA=0`: Skip CUDA extension build during installation
- `SAM2_BUILD_ALLOW_ERRORS=0`: Force stopping on CUDA build errors
- `CUDA_HOME`: Specify CUDA toolkit path if not automatically detected

## Architecture Overview

### Core Components

**Model Architecture** (`sam2/modeling/`):
- `sam2_base.py`: Base SAM2 model implementation
- `backbones/`: Image encoder with Hiera backbone architecture
- `memory_attention.py`: Memory attention mechanism for video processing
- `memory_encoder.py`: Encodes masks and features into memory
- `sam/`: SAM-specific components (mask decoder, prompt encoder, transformer)

**Predictors** (`sam2/`):
- `sam2_image_predictor.py`: Static image segmentation interface
- `sam2_video_predictor.py`: Video segmentation with temporal consistency
- `automatic_mask_generator.py`: Automatic mask generation for images

**Model Building** (`sam2/build_sam.py`):
- `build_sam2()`: Build image predictor
- `build_sam2_video_predictor()`: Build video predictor
- `build_sam2_hf()`: Load from Hugging Face Hub

### Configuration System

Uses Hydra for configuration management with YAML files in `sam2_configs/`:
- `sam2_hiera_t.yaml`: Tiny model (38.9M parameters)
- `sam2_hiera_s.yaml`: Small model (46M parameters)  
- `sam2_hiera_b+.yaml`: Base Plus model (80.8M parameters)
- `sam2_hiera_l.yaml`: Large model (224.4M parameters)

### Custom Pipeline Implementations

**Pipeline Implementation** (`pipeline_implementation/`):
- SnakeMake-based video processing pipelines
- Integration with DeepLabCut for body part tracking
- Batch processing capabilities for large video datasets
- Server deployment configurations

**SAM2 SnakeMake Scripts** (`SAM2_snakemake_scripts/`):
- Video processing from JPEG frames
- Error correction pipelines for DLC integration
- Debug and batch processing variants

## Usage Patterns

### Image Segmentation
```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
```

### Video Segmentation
```python
from sam2.build_sam import build_sam2_video_predictor

checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)
```

### Hugging Face Integration
Models can be loaded directly from Hugging Face:
```python
from sam2.sam2_image_predictor import SAM2ImagePredictor
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
```

## Key Dependencies

Core requirements: `torch>=2.3.1`, `torchvision>=0.18.1`, `numpy>=1.24.4`, `hydra-core>=1.3.2`

Demo dependencies: `matplotlib>=3.9.1`, `jupyter>=1.0.0`, `opencv-python>=4.7.0`

## Important Notes

- CUDA extension compilation is optional but enables post-processing features
- The repository includes extensive pipeline implementations for batch video processing
- Model checkpoints are not included in the repository and must be downloaded separately
- Configuration uses Hydra's compose API with YAML config files
- Video processing maintains temporal consistency through memory attention mechanisms