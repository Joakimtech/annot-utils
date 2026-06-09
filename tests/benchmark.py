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
        
        # 100 images, ~1000 annotations (reduced for benchmark speed)
        num_images = 100
        annotations_per_image = 10
        
        for img_id in range(num_images):
            data["images"].append({
                "id": img_id,
                "file_name": f"img_{img_id}.jpg",
                "width": 1920,
                "height": 1080
            })
            
            for ann_id in range(annotations_per_image):
                data["annotations"].append({
                    "id": len(data["annotations"]),  # Unique ID
                    "image_id": img_id,
                    "category_id": ann_id % 10 + 1,  # COCO uses 1-indexed
                    "bbox": [100 * (ann_id % 5), 100 * (ann_id % 4), 200, 150],
                    "area": 30000,
                    "iscrowd": 0
                })
        
        expected_total = num_images * annotations_per_image
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            return f.name, expected_total
    
    def test_conversion_performance(self, benchmark):
        """Benchmark COCO to YOLO conversion"""
        coco_path, expected_total = self.large_coco_dataset()
        converter = AnnotationConverter(verbose=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = benchmark(converter.coco_to_yolo, coco_path, tmpdir)
            
            # Fix: Check different possible return structures
            if isinstance(result, dict):
                actual_total = result.get('total_annotations', 0)
            elif isinstance(result, (int, float)):
                actual_total = result
            else:
                actual_total = len(result) if hasattr(result, '__len__') else 0
            
            # Fix: Use assert with helpful message
            assert actual_total == expected_total, \
                f"Expected {expected_total} annotations, but got {actual_total}"
            
            # Fix: Also check that files were created
            output_dir = Path(tmpdir)
            txt_files = list(output_dir.glob('*.txt'))
            assert len(txt_files) > 0, "No YOLO files were created"
    
    def test_validation_performance(self, benchmark):
        """Benchmark validation performance"""
        coco_path, expected_total = self.large_coco_dataset()
        validator = AnnotationValidator()
        
        # Create mock image directory with dummy images
        with tempfile.TemporaryDirectory() as img_tmpdir:
            # Create dummy images to avoid "image not found" warnings
            from PIL import Image
            
            for img_id in range(100):  # Match number of images
                img_path = Path(img_tmpdir) / f"img_{img_id}.jpg"
                # Create a blank image
                img = Image.new('RGB', (1920, 1080), color='white')
                img.save(img_path)
            
            # Run validation
            report = benchmark(validator.validate_coco, coco_path, img_tmpdir)
            
            # Fix: Check report attributes correctly
            assert report.total_annotations == expected_total, \
                f"Expected {expected_total} annotations in report, got {report.total_annotations}"
            
            # Fix: Quality score should be reasonable (not NaN)
            assert 0 <= report.quality_score <= 100, \
                f"Quality score {report.quality_score} out of range"
    
    def test_conversion_consistency(self):
        """Test that conversion back and forth preserves data"""
        coco_path, expected_total = self.large_coco_dataset()
        converter = AnnotationConverter(verbose=False)
        
        # COCO → YOLO
        with tempfile.TemporaryDirectory() as yolo_dir:
            result1 = converter.coco_to_yolo(coco_path, yolo_dir)
            
            # Fix: Handle different return types
            if isinstance(result1, dict):
                assert result1.get('total_annotations', 0) == expected_total
            else:
                # If result is something else, at least check files exist
                txt_files = list(Path(yolo_dir).glob('*.txt'))
                assert len(txt_files) > 0


# Fix: Add a separate test for empty annotations
class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_annotations(self):
        """Test handling of empty annotation file"""
        empty_coco = {
            "images": [{"id": 1, "file_name": "test.jpg", "width": 800, "height": 600}],
            "annotations": [],
            "categories": [{"id": 1, "name": "test"}]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(empty_coco, f)
            empty_path = f.name
        
        converter = AnnotationConverter(verbose=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = converter.coco_to_yolo(empty_path, tmpdir)
            
            # Fix: Handle both dict and other return types
            if isinstance(result, dict):
                assert result.get('total_annotations', 0) == 0
                assert result.get('images_with_annotations', 0) == 0
            else:
                # If no annotations, no txt files should be created (or empty ones)
                txt_files = list(Path(tmpdir).glob('*.txt'))
                # Either no files or files with no content
                for txt_file in txt_files:
                    assert txt_file.stat().st_size == 0
    
    def test_missing_image_directory(self):
        """Test validation with missing images"""
        coco_path, _ = self.large_coco_dataset()
        validator = AnnotationValidator()
        
        with tempfile.TemporaryDirectory() as empty_dir:
            # Run validation with missing images (should handle gracefully)
            report = validator.validate_coco(coco_path, empty_dir)
            
            # Fix: Should still have warnings but not crash
            assert report is not None
            assert len(report.warnings) > 0, "Should warn about missing images"
