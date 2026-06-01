
import json
import tempfile
from pathlib import Path
from annot_utils.converter import AnnotationConverter

def create_sample_coco():
    """Create a sample COCO JSON file for testing"""
    sample_coco = {
        "images": [
            {"id": 1, "file_name": "test_image.jpg", "width": 800, "height": 600}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 150, 200, 300],
                "area": 60000,
                "iscrowd": 0
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [400, 200, 150, 250],
                "area": 37500,
                "iscrowd": 0
            }
        ],
        "categories": [
            {"id": 1, "name": "cat", "supercategory": "animal"},
            {"id": 2, "name": "dog", "supercategory": "animal"}
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_coco, f)
        return f.name

def test_converter():
    """Test both conversion directions"""
    print(" Testing Annotation Converter\n")
    
    converter = AnnotationConverter(verbose=True)
    
    # Test COCO → YOLO
    print("\n1. Testing COCO → YOLO conversion:")
    coco_file = create_sample_coco()
    output_dir = tempfile.mkdtemp()
    
    stats = converter.coco_to_yolo(coco_file, output_dir)
    assert stats['total_images'] == 1
    assert stats['total_annotations'] == 2
    print("   ✓ COCO → YOLO passed")
    
    # Note: For full test, you'd need actual image files
    print("\n⚠️ Full YOLO → COCO test requires image files")
    print(" Converter tests completed")

if __name__ == "__main__":
    test_converter()
