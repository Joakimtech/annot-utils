"""
Simplified performance benchmarks for annot-utils
"""

import pytest
import tempfile
import json
from pathlib import Path


class TestBenchmarks:
    """Simplified benchmark tests"""
    
    @pytest.fixture
    def sample_coco_data(self):
        """Generate a small sample COCO dataset"""
        data = {
            "images": [
                {"id": 1, "file_name": "test1.jpg", "width": 800, "height": 600},
                {"id": 2, "file_name": "test2.jpg", "width": 800, "height": 600}
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 200, 150], "area": 30000, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 1, "bbox": [400, 200, 150, 150], "area": 22500, "iscrowd": 0},
                {"id": 3, "image_id": 2, "category_id": 2, "bbox": [300, 250, 180, 200], "area": 36000, "iscrowd": 0}
            ],
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            return f.name, len(data['annotations'])
    
    def test_conversion_basic(self, sample_coco_data):
        """Basic test of conversion functionality"""
        from annot_utils.converter import AnnotationConverter
        
        coco_path, expected_count = sample_coco_data
        converter = AnnotationConverter(verbose=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = converter.coco_to_yolo(coco_path, tmpdir)
            
            # Check that conversion completed
            assert result is not None
            
            # Check that output files were created
            output_dir = Path(tmpdir)
            txt_files = list(output_dir.glob('*.txt'))
            assert len(txt_files) == 2  # One for each image
            
            # Check class mapping file was created
            mapping_file = output_dir / "class_mapping.json"
            assert mapping_file.exists()
    
    def test_validation_basic(self, sample_coco_data):
        """Basic test of validation functionality"""
        from annot_utils.validator import AnnotationValidator
        
        coco_path, expected_count = sample_coco_data
        validator = AnnotationValidator()
        
        # Create dummy images
        with tempfile.TemporaryDirectory() as img_dir:
            from PIL import Image
            
            # Create dummy images
            for i in range(1, 3):
                img = Image.new('RGB', (800, 600), color='white')
                img.save(Path(img_dir) / f"test{i}.jpg")
            
            # Run validation
            report = validator.validate_coco(coco_path, img_dir)
            
            # Basic assertions
            assert report.total_annotations == expected_count
            assert hasattr(report, 'quality_score')
            assert hasattr(report, 'errors')
            assert hasattr(report, 'warnings')
