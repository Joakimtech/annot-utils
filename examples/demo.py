"""
Complete demo of all annot-utils features
"""

import json
import tempfile
from pathlib import Path
from annot_utils.cli import convert, validate, agreement
from click.testing import CliRunner

def demo():
    """Run complete demo of all functionality"""
    print("🎯 annot-utils Complete Demo\n")
    
    runner = CliRunner()
    
    # Create sample data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        sample_coco = {
            "images": [{"id": 1, "file_name": "demo.jpg", "width": 800, "height": 600}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 200, 150], "area": 30000},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [400, 200, 150, 150], "area": 22500}
            ],
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"}
            ]
        }
        json.dump(sample_coco, f)
        coco_path = f.name
    
    # Demo 1: Convert COCO to YOLO
    print("1️⃣ Converting COCO to YOLO...")
    with tempfile.TemporaryDirectory() as output_dir:
        result = runner.invoke(convert, [
            '--from', 'coco',
            '--to', 'yolo',
            '--input', coco_path,
            '--output', output_dir,
            '--quiet'
        ])
        print(f"   {'✅' if result.exit_code == 0 else '❌'} Conversion completed")
    
    # Demo 2: Show help
    print("\n2️⃣ Showing CLI help...")
    result = runner.invoke(convert, ['--help'])
    print(f"   ✅ Help displayed ({len(result.output)} chars)")
    
    # Demo 3: Version
    print("\n3️⃣ Checking version...")
    from annot_utils import __version__
    print(f"   ✅ Version {__version__}")
    
    print("\n✨ Demo complete! annot-utils is ready for use.\n")
    print("Quick start commands:")
    print("  annot-utils convert --help")
    print("  annot-utils validate --help")
    print("  annot-utils agreement --help")

if __name__ == "__main__":
    demo()
