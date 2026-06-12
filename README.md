# annot-utils
A Python CLI tool for converting between annotation formats (COCO ↔ YOLO), validating label quality, and calculating inter-annotator agreement. Perfect for data annotation teams and computer vision workflows


[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/joakimtech/annot-utils/workflows/Tests/badge.svg)](https://github.com/joakimtech/annot-utils/actions)
[![PyPI version](https://badge.fury.io/py/annot-utils.svg)](https://badge.fury.io/py/annot-utils)

A professional command-line toolkit for computer vision annotation workflows. Convert between annotation formats, validate label quality, measure inter-annotator agreement, and generate comprehensive statistics.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Commands Reference](#commands-reference)
  - [Convert](#convert)
  - [Validate](#validate)
  - [Agreement](#agreement)
  - [Statistics](#statistics)
- [Use Cases](#use-cases)
- [API Reference](#api-reference)
- [Benchmarks](#benchmarks)
- [Contributing](#contributing)
- [License](#license)

## Overview

Data annotation is a critical step in computer vision projects, but working with different annotation formats and ensuring quality can be challenging. annot-utils solves these problems by providing:

- Bidirectional conversion between COCO JSON and YOLO txt formats
- Comprehensive validation to catch annotation errors before training
- Statistical analysis to measure inter-annotator consistency
- Detailed reporting for quality assurance

The tool is designed for machine learning engineers, data scientists, and annotation teams who need reliable, automated annotation processing.

## Features

| Feature | Description |
|---------|-------------|
| Format Conversion | Convert between COCO and YOLO formats while preserving all annotation data |
| Batch Processing | Process entire datasets with a single command |
| Validation | Check for boundary violations, size issues, aspect ratio problems, and overlaps |
| HTML Reports | Generate shareable validation reports with visual summaries |
| Agreement Analysis | Calculate Cohen's Kappa, precision, recall, F1 scores, and mean IoU between annotators |
| Statistics Generation | Produce detailed dataset statistics including class distributions and box metrics |
| Cross-Platform | Works on Linux, Windows, and macOS |
| Docker Support | Run in containerized environments for reproducibility |

## Installation

### From PyPI

```bash
pip install annot-utils
```

### From Source

```bash
git clone https://github.com/joakimtech/annot-utils.git
cd annot-utils
pip install -e .
```

### Using Docker

```bash
docker pull joakimtech/annot-utils:latest
docker run joakimtech/annot-utils --help
```

### Verification

```bash
annot-utils --version
annot-utils --help
```

## Quick Start

Convert COCO annotations to YOLO format:

```bash
annot-utils convert --from coco --to yolo --input annotations.json --output ./yolo_labels/
```

Validate annotations before training:

```bash
annot-utils validate --format coco --annotations labels.json --images ./images/
```

Measure agreement between two annotators:

```bash
annot-utils agreement --format coco --annotator1 annotator_a.json --annotator2 annotator_b.json
```

## Commands Reference

### Convert

Convert annotations between COCO and YOLO formats.

#### COCO to YOLO

```bash
annot-utils convert \
  --from coco \
  --to yolo \
  --input annotations.json \
  --output ./yolo_annotations/ \
  --quiet
```

**Output:**
- Individual txt files for each image in YOLO format
- class_mapping.json file mapping class IDs to names

#### YOLO to COCO

```bash
annot-utils convert \
  --from yolo \
  --to coco \
  --input ./yolo_annotations/ \
  --output coco_annotations.json \
  --image-dir ./images/ \
  --class-names cat,dog,bird
```

**Class Names Options:**
- Comma-separated list: `--class-names cat,dog,bird`
- File with one class per line: `--class-names classes.txt`

#### Batch Conversion

Convert multiple COCO files in a directory:

```bash
for file in ./coco_annotations/*.json; do
  annot-utils convert \
    --from coco --to yolo \
    --input "$file" \
    --output "./yolo_${file%.json}/"
done
```

### Validate

Check annotation quality and integrity.

#### Basic Validation

```bash
annot-utils validate \
  --format coco \
  --annotations annotations.json \
  --images ./images/ \
  --output validation_report.html
```

#### Validation with Custom Thresholds

```bash
annot-utils validate \
  --format coco \
  --annotations labels.json \
  --images ./images/ \
  --min-box-size 500 \
  --max-aspect-ratio 5.0 \
  --strict
```

#### Validation Checks Performed

| Check | Description |
|-------|-------------|
| Boundary | Box coordinates within image dimensions |
| Size | Minimum and maximum box area limits |
| Aspect Ratio | Width/height ratio within acceptable range |
| Overlap | Significant overlap detection between boxes |
| File Existence | Image files exist for all annotations |
| Category Validity | Category IDs match defined classes |

#### Validation Report Output

The validation command generates a detailed report including:

- Total images and annotations processed
- Count of errors, warnings, and informational messages
- Quality score from 0 to 100
- Distribution of box sizes and aspect ratios
- List of all issues with locations
- HTML visualization for easy sharing

### Agreement

Calculate inter-annotator agreement metrics.

#### COCO Format Agreement

```bash
annot-utils agreement \
  --format coco \
  --annotator1 annotator_A.json \
  --annotator2 annotator_B.json \
  --output agreement_report.txt \
  --iou-threshold 0.5
```

#### YOLO Format Agreement

```bash
annot-utils agreement \
  --format yolo \
  --annotator1 ./user1_labels/ \
  --annotator2 ./user2_labels/ \
  --image-dir ./images/ \
  --class-names classes.txt \
  --iou-threshold 0.5
```

#### Agreement Metrics Explained

| Metric | Description | Range |
|--------|-------------|-------|
| Cohen's Kappa | Agreement beyond chance | -1 to 1 |
| Precision | Matched boxes / Annotator 1 total | 0 to 1 |
| Recall | Matched boxes / Annotator 2 total | 0 to 1 |
| F1 Score | Harmonic mean of precision and recall | 0 to 1 |
| Mean IoU | Average overlap of matched boxes | 0 to 1 |

#### Kappa Interpretation Guide

| Kappa Value | Agreement Level |
|-------------|-----------------|
| 0.81 - 1.00 | Almost perfect |
| 0.61 - 0.80 | Substantial |
| 0.41 - 0.60 | Moderate |
| 0.21 - 0.40 | Fair |
| 0.00 - 0.20 | Slight |

### Statistics

Generate comprehensive dataset statistics.

```bash
annot-utils stats \
  --format coco \
  --input annotations.json \
  --output stats.json
```

#### Statistics Output Example

```json
{
  "format": "COCO",
  "total_images": 150,
  "total_annotations": 2340,
  "total_categories": 5,
  "categories": {
    "cat": 845,
    "dog": 723,
    "bird": 412,
    "car": 210,
    "person": 150
  },
  "bbox_statistics": {
    "mean_area": 12450.5,
    "std_area": 3420.2,
    "min_area": 450,
    "max_area": 125000,
    "mean_aspect_ratio": 1.23
  }
}
```

## Use Cases

### 1. Data Preparation for Model Training

Convert annotation formats to match your training pipeline requirements:

```bash
annot-utils convert --from coco --to yolo --input custom.json --output ./yolo/
```

### 2. Quality Assurance Before Training

Validate annotations to prevent wasted compute resources on bad data:

```bash
annot-utils validate --format yolo --annotations ./labels/ --images ./images/ --strict
```

### 3. Annotator Performance Evaluation

Measure consistency between team members:

```bash
annot-utils agreement --annotator1 junior.json --annotator2 senior.json
```

### 4. Dataset Analysis and Understanding

Analyze class distributions and box statistics:

```bash
annot-utils stats --input dataset.json --output analysis.json
```

### 5. Continuous Integration for Annotation Quality

Integrate validation into CI/CD pipelines:

```yaml
# GitHub Actions example
- name: Validate Annotations
  run: |
    annot-utils validate \
      --format coco \
      --annotations $ANNOTATION_PATH \
      --images ./images/ \
      --strict
```

## API Reference

### Python API Usage

```python
from annot_utils import AnnotationConverter, AnnotationValidator, AgreementCalculator

# Convert annotations
converter = AnnotationConverter(verbose=True)
stats = converter.coco_to_yolo("annotations.json", "./yolo_output/")

# Validate annotations
validator = AnnotationValidator(min_box_size=100, max_aspect_ratio=10.0)
report = validator.validate_coco("annotations.json", "./images/")
print(f"Quality Score: {report.quality_score}/100")

# Calculate agreement
calculator = AgreementCalculator(iou_threshold=0.5)
metrics = calculator.calculate_coco_agreement("annotator1.json", "annotator2.json")
print(metrics.summary())
```

### Configuration Options

#### AnnotationValidator Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| min_box_size | 100 | Minimum allowed box area in pixels |
| max_box_size_ratio | 0.95 | Maximum box area as fraction of image |
| max_aspect_ratio | 10.0 | Maximum width/height ratio allowed |
| min_confidence | 0.0 | Minimum confidence score threshold |

#### AgreementCalculator Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| iou_threshold | 0.5 | Minimum IoU for box matching |

## Benchmarks

Performance results on standard hardware (Intel i7-10750H, 16GB RAM):

| Dataset Size | Conversion Time | Validation Time | Agreement Time |
|-------------|----------------|-----------------|----------------|
| 100 images (1k boxes) | 0.3s | 0.5s | 0.2s |
| 1,000 images (10k boxes) | 2.1s | 3.2s | 1.8s |
| 10,000 images (100k boxes) | 18s | 28s | 15s |

## Platform Support

| Operating System | Python Versions | Architecture |
|-----------------|-----------------|--------------|
| Ubuntu Linux (20.04, 22.04, 24.04) | 3.8 - 3.12 | x86_64, ARM64 |
| Windows (10, 11, Server 2022) | 3.8 - 3.12 | x86_64 |
| macOS (Monterey, Ventura, Sonoma) | 3.8 - 3.12 | x86_64, ARM64 |

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies (`pip install -e .[dev]`)
4. Make your changes
5. Run tests (`pytest tests/`)
6. Format code (`black annot_utils/`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/joakimtech/annot-utils.git
cd annot-utils
pip install -e .[dev]
pytest tests/
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=annot_utils --cov-report=html

# Run specific test file
pytest tests/test_converter.py
```

## Docker Support

Build the Docker image:

```bash
docker build -t annot-utils .
```

Run in container:

```bash
docker run annot-utils convert --from coco --to yolo --input /data/annotations.json --output /output/
```

## Troubleshooting

### Common Issues

**Issue: Module not found after installation**
```bash
pip install --upgrade annot-utils
```

**Issue: Permission denied when running commands**
```bash
# On Unix/macOS
chmod +x /path/to/annot-utils

# On Windows, run as administrator
```

**Issue: Images not found during validation**
- Ensure --image-dir path is correct
- Check that image filenames match those in annotation files
- Supported formats: .jpg, .jpeg, .png, .bmp

## Roadmap

Future enhancements planned:

- Pascal VOC format support
- CVAT XML format support
- Rotated bounding boxes (OBB)
- Polygon annotation support
- Web-based visualization dashboard
- Parallel processing for large datasets
- Label Studio integration

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

Joakim Tech - [GitHub](https://github.com/joakimtech)

## Acknowledgments

- COCO API for annotation format specification
- Shapely library for geometric computations
- Click framework for CLI interface
- Open source community for testing and contributions

## Support

- Documentation: [GitHub Repository](https://github.com/joakimtech/annot-utils)
- Issue Tracker: [GitHub Issues](https://github.com/joakimtech/annot-utils/issues)
- Discussions: [GitHub Discussions](https://github.com/joakimtech/annot-utils/discussions)
```

