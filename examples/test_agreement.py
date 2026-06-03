"""
Test script for inter-annotator agreement calculator
"""

import json
import tempfile
from pathlib import Path
from annot_utils.agreement import AgreementCalculator

def create_test_coco_annotator1():
    """Create first annotator's COCO file"""
    return {
        "images": [
            {"id": 1, "file_name": "img1.jpg", "width": 800, "height": 600},
            {"id": 2, "file_name": "img2.jpg", "width": 800, "height": 600}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 200, 150], "area": 30000},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [400, 200, 150, 150], "area": 22500},
            {"id": 3, "image_id": 2, "category_id": 2, "bbox": [300, 250, 180, 200], "area": 36000}
        ],
        "categories": [
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "dog"}
        ]
    }

def create_test_coco_annotator2():
    """Create second annotator's COCO file (slightly different)"""
    return {
        "images": [
            {"id": 1, "file_name": "img1.jpg", "width": 800, "height": 600},
            {"id": 2, "file_name": "img2.jpg", "width": 800, "height": 600}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [110, 95, 195, 155], "area": 30225},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [410, 210, 140, 140], "area": 19600},
            {"id": 3, "image_id": 2, "category_id": 2, "bbox": [305, 255, 175, 195], "area": 34125},
            {"id": 4, "image_id": 2, "category_id": 1, "bbox": [500, 100, 100, 100], "area": 10000}  # Extra box
        ],
        "categories": [
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "dog"}
        ]
    }

def main():
    print("🧪 Testing Inter-Annotator Agreement Calculator\n")
    
    # Create test COCO files
    coco1 = create_test_coco_annotator1()
    coco2 = create_test_coco_annotator2()
    
    coco1_path = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    coco2_path = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    
    json.dump(coco1, coco1_path)
    json.dump(coco2, coco2_path)
    coco1_path.close()
    coco2_path.close()
    
    # Initialize calculator
    calculator = AgreementCalculator(iou_threshold=0.5)
    
    # Calculate agreement
    print("1. Calculating agreement between two annotators...")
    metrics = calculator.calculate_coco_agreement(coco1_path.name, coco2_path.name)
    
    # Display results
    print(metrics.summary())
    
    print("\n2. Per-class metrics:")
    for class_name, class_metrics in metrics.per_class_metrics.items():
        print(f"\n   {class_name.upper()}:")
        print(f"   - Precision: {class_metrics['precision']:.3f}")
        print(f"   - Recall: {class_metrics['recall']:.3f}")
        print(f"   - F1 Score: {class_metrics['f1_score']:.3f}")
        print(f"   - Mean IoU: {class_metrics['mean_iou']:.3f}")
    
    # Generate report
    report_path = tempfile.NamedTemporaryFile(suffix='.txt', delete=False).name
    calculator.generate_agreement_report(metrics, report_path)
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    print(f"📊 JSON data saved to: {report_path.replace('.txt', '.json')}")
    
    # Interpret results
    print("\n📈 Interpretation:")
    if metrics.f1_score >= 0.8:
        print("   ✅ Excellent agreement - Annotators are highly consistent")
    elif metrics.f1_score >= 0.6:
        print("   👍 Good agreement - Annotations are reliable")
    elif metrics.f1_score >= 0.4:
        print("   ⚠️ Moderate agreement - Some inconsistencies need review")
    else:
        print("   ❌ Poor agreement - Major discrepancies found")
    
    if metrics.kappa > 0.6:
        print("   ✅ Strong categorical agreement beyond chance")
    elif metrics.kappa > 0.4:
        print("   ⚠️ Moderate categorical agreement")
    else:
        print("   ❌ Low categorical agreement - Review labeling guidelines")
    
    print("\n✅ Agreement calculator test completed!")

if __name__ == "__main__":
    main()
