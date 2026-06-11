"""
Performance benchmarks for annot-utils
"""

import pytest
import tempfile
import json
from pathlib import Path
from annot_utils.converter import AnnotationConverter
from annot_utils.validator import AnnotationValidator


class TestBenchmarks:
    """Benchmark various operations"""
    
    @pytest.fixture
    def large_coco_dataset(self):
        """Generate a large synthetic COCO dataset"""
        data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": f"class_{i}"} for i in range(10)]
        }
        
        # 1000 images, ~10k annotations
        for img_id in range(1000):
            data["images"].append({
                "id": img_id,
                "file_name": f"img_{img_id}.jpg",
                "width": 1920,
                "height": 1080
            })
            
            for ann_id in range(10):
                data["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": ann_id % 10,
                    "bbox": [100, 100, 200, 150],
                    "area": 30000,
                    "iscrowd": 0
                })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            return f.name
    
    def test_conversion_performance(self, benchmark, large_coco_dataset):
        """Benchmark COCO to YOLO conversion"""
        converter = AnnotationConverter(verbose=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = benchmark(converter.coco_to_yolo, large_coco_dataset, tmpdir)
            assert result['total_annotations'] == 10000
    
    def test_validation_performance(self, benchmark, large_coco_dataset):
        """Benchmark validation performance"""
        validator = AnnotationValidator()
        
        # Mock image directory (just for structure)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = benchmark(validator.validate_coco, large_coco_dataset, tmpdir)
            assert result.total_annotations == 10000
