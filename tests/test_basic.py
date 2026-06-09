"""
Basic tests that should always pass
"""

import pytest
import tempfile
import json
from pathlib import Path


class TestBasicFunctionality:
    """Basic functionality tests"""
    
    def test_imports(self):
        """Test that all modules import correctly"""
        from annot_utils import AnnotationConverter, AnnotationValidator, AgreementCalculator
        from annot_utils.converter import AnnotationConverter
        from annot_utils.validator import AnnotationValidator
        from annot_utils.agreement import AgreementCalculator
        assert True
    
    def test_version(self):
        """Test that version is set"""
        from annot_utils import __version__
        assert __version__ is not None
        assert isinstance(__version__, str)
    
    def test_converter_initialization(self):
        """Test converter initialization"""
        from annot_utils.converter import AnnotationConverter
        converter = AnnotationConverter(verbose=False)
        assert converter is not None
    
    def test_validator_initialization(self):
        """Test validator initialization"""
        from annot_utils.validator import AnnotationValidator
        validator = AnnotationValidator()
        assert validator is not None
    
    def test_agreement_initialization(self):
        """Test agreement calculator initialization"""
        from annot_utils.agreement import AgreementCalculator
        calculator = AgreementCalculator()
        assert calculator is not None
    
    def test_sample_conversion(self):
        """Test conversion with minimal data"""
        from annot_utils.converter import AnnotationConverter
        
        # Create minimal COCO data
        sample_coco = {
            "images": [{"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50], "area": 2500, "iscrowd": 0}
            ],
            "categories": [{"id": 1, "name": "test"}]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_coco, f)
            coco_path = f.name
        
        converter = AnnotationConverter(verbose=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # This should not raise an exception
            result = converter.coco_to_yolo(coco_path, tmpdir)
            assert result is not None
    
    def test_cli_help(self):
        """Test CLI help command works"""
        import subprocess
        import sys
        
        result = subprocess.run(
            [sys.executable, '-m', 'annot_utils.cli', '--help'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'Usage' in result.stdout or 'Usage' in result.stderr


class TestEdgeCases:
    """Test edge cases"""
    
    def test_empty_coco(self):
        """Test handling of empty COCO file"""
        from annot_utils.converter import AnnotationConverter
        
        empty_coco = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(empty_coco, f)
            coco_path = f.name
        
        converter = AnnotationConverter(verbose=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should handle empty file gracefully
            result = converter.coco_to_yolo(coco_path, tmpdir)
            assert result is not None
