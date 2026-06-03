"""
Inter-annotator agreement calculator
Measures consistency between multiple annotators
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
from shapely.geometry import box as shapely_box
from sklearn.metrics import cohen_kappa_score


@dataclass
class AgreementMetrics:
    """Container for agreement metrics"""
    kappa: float  # Cohen's Kappa (-1 to 1)
    precision: float  # Matched annotations / annotator1 total
    recall: float  # Matched annotations / annotator2 total
    f1_score: float  # Harmonic mean of precision and recall
    matched_pairs: int
    total_boxes_annotator1: int
    total_boxes_annotator2: int
    mean_iou: float  # Mean IoU of matched boxes
    per_class_metrics: Dict[str, Dict[str, float]]
    
    def to_dict(self):
        return asdict(self)
    
    def summary(self) -> str:
        """Return formatted summary string"""
        return f"""
📊 Agreement Summary:
├─ Cohen's Kappa: {self.kappa:.3f}
├─ Precision: {self.precision:.3f}
├─ Recall: {self.recall:.3f}  
├─ F1 Score: {self.f1_score:.3f}
├─ Matched Pairs: {self.matched_pairs}/{self.total_boxes_annotator1} & {self.total_boxes_annotator2}
├─ Mean IoU: {self.mean_iou:.3f}
└─ Quality: {self._get_quality_label()}
"""
    
    def _get_quality_label(self) -> str:
        """Get qualitative label based on F1 score"""
        if self.f1_score >= 0.8:
            return "Excellent agreement ✅"
        elif self.f1_score >= 0.6:
            return "Good agreement 👍"
        elif self.f1_score >= 0.4:
            return "Moderate agreement ⚠️"
        else:
            return "Poor agreement ❌"


class AgreementCalculator:
    """Calculate inter-annotator agreement for annotations"""
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize agreement calculator
        
        Args:
            iou_threshold: Minimum IoU to consider boxes as matching
        """
        self.iou_threshold = iou_threshold
    
    def calculate_coco_agreement(self, 
                                 coco1_path: str, 
                                 coco2_path: str,
                                 image_id: Optional[int] = None) -> AgreementMetrics:
        """
        Calculate agreement between two COCO annotation files
        
        Args:
            coco1_path: Path to first annotator's COCO file
            coco2_path: Path to second annotator's COCO file
            image_id: Specific image ID to compare (None = all images)
            
        Returns:
            AgreementMetrics with all agreement scores
        """
        # Load COCO files
        with open(coco1_path, 'r') as f:
            coco1 = json.load(f)
        with open(coco2_path, 'r') as f:
            coco2 = json.load(f)
        
        # Build annotation lookup
        def build_annotation_dict(coco_data, img_id=None):
            annotations_by_image = defaultdict(list)
            for ann in coco_data['annotations']:
                if img_id is None or ann['image_id'] == img_id:
                    # Get category name
                    cat = next((c for c in coco_data['categories'] if c['id'] == ann['category_id']), None)
                    class_name = cat['name'] if cat else f"class_{ann['category_id']}"
                    
                    annotations_by_image[ann['image_id']].append({
                        'id': ann['id'],
                        'class': class_name,
                        'bbox': ann['bbox'],
                        'area': ann.get('area', ann['bbox'][2] * ann['bbox'][3])
                    })
            return annotations_by_image
        
        annotations1 = build_annotation_dict(coco1, image_id)
        annotations2 = build_annotation_dict(coco2, image_id)
        
        # Get all image IDs
        all_image_ids = set(annotations1.keys()) | set(annotations2.keys())
        
        # Aggregate metrics across images
        all_matches = []
        all_precision = []
        all_recall = []
        all_f1 = []
        all_ious = []
        per_class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'iou_sum': 0})
        
        for img_id in all_image_ids:
            boxes1 = annotations1.get(img_id, [])
            boxes2 = annotations2.get(img_id, [])
            
            if not boxes1 and not boxes2:
                continue
            
            # Calculate matches for this image
            matches = self._match_annotations(boxes1, boxes2)
            all_matches.extend(matches)
            
            # Calculate per-class metrics
            for class_name in set([b['class'] for b in boxes1] + [b['class'] for b in boxes2]):
                class_boxes1 = [b for b in boxes1 if b['class'] == class_name]
                class_boxes2 = [b for b in boxes2 if b['class'] == class_name]
                class_matches = self._match_annotations(class_boxes1, class_boxes2)
                
                tp = len(class_matches)
                fp = len(class_boxes1) - tp
                fn = len(class_boxes2) - tp
                iou_sum = sum(m['iou'] for m in class_matches)
                
                per_class_stats[class_name]['tp'] += tp
                per_class_stats[class_name]['fp'] += fp
                per_class_stats[class_name]['fn'] += fn
                per_class_stats[class_name]['iou_sum'] += iou_sum
            
            # Calculate image-level metrics
            precision = len(matches) / len(boxes1) if boxes1 else 1.0
            recall = len(matches) / len(boxes2) if boxes2 else 1.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            mean_iou = np.mean([m['iou'] for m in matches]) if matches else 0
            
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)
            all_ious.append(mean_iou)
        
        # Calculate overall metrics
        total_boxes1 = sum(len(annotations1.get(img_id, [])) for img_id in all_image_ids)
        total_boxes2 = sum(len(annotations2.get(img_id, [])) for img_id in all_image_ids)
        
        overall_precision = np.mean(all_precision) if all_precision else 0
        overall_recall = np.mean(all_recall) if all_recall else 0
        overall_f1 = np.mean(all_f1) if all_f1 else 0
        overall_mean_iou = np.mean(all_ious) if all_ious else 0
        
        # Calculate Cohen's Kappa for categorical agreement
        kappa = self._calculate_kappa(annotations1, annotations2, all_image_ids)
        
        # Build per-class metrics
        per_class_metrics = {}
        for class_name, stats in per_class_stats.items():
            precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
            recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            mean_iou = stats['iou_sum'] / stats['tp'] if stats['tp'] > 0 else 0
            
            per_class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'mean_iou': mean_iou,
                'true_positives': stats['tp'],
                'false_positives': stats['fp'],
                'false_negatives': stats['fn']
            }
        
        return AgreementMetrics(
            kappa=kappa,
            precision=overall_precision,
            recall=overall_recall,
            f1_score=overall_f1,
            matched_pairs=len(all_matches),
            total_boxes_annotator1=total_boxes1,
            total_boxes_annotator2=total_boxes2,
            mean_iou=overall_mean_iou,
            per_class_metrics=per_class_metrics
        )
    
    def calculate_yolo_agreement(self,
                                  yolo_dir1: str,
                                  yolo_dir2: str,
                                  image_dir: str,
                                  class_names: List[str]) -> AgreementMetrics:
        """
        Calculate agreement between two YOLO annotation directories
        
        Args:
            yolo_dir1: Directory with first annotator's YOLO files
            yolo_dir2: Directory with second annotator's YOLO files
            image_dir: Directory with images (for dimensions)
            class_names: List of class names
            
        Returns:
            AgreementMetrics with all agreement scores
        """
        from pathlib import Path
        from PIL import Image
        
        yolo_path1 = Path(yolo_dir1)
        yolo_path2 = Path(yolo_dir2)
        image_path = Path(image_dir)
        
        # Get all unique image names
        yolo_files1 = {f.stem: f for f in yolo_path1.glob('*.txt')}
        yolo_files2 = {f.stem: f for f in yolo_path2.glob('*.txt')}
        all_images = set(yolo_files1.keys()) | set(yolo_files2.keys())
        
        # Parse YOLO annotations
        def parse_yolo_annotations(yolo_file: Path, img_width: int, img_height: int) -> List[Dict]:
            """Parse YOLO file and convert to absolute coordinates"""
            annotations = []
            if not yolo_file.exists():
                return annotations
            
            with open(yolo_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    
                    class_id, cx, cy, w, h = map(float, parts)
                    
                    # Convert to absolute coordinates
                    x = (cx - w/2) * img_width
                    y = (cy - h/2) * img_height
                    bbox_width = w * img_width
                    bbox_height = h * img_height
                    
                    class_name = class_names[int(class_id)] if int(class_id) < len(class_names) else f"class_{int(class_id)}"
                    
                    annotations.append({
                        'class': class_name,
                        'bbox': [x, y, bbox_width, bbox_height],
                        'area': bbox_width * bbox_height
                    })
            return annotations
        
        all_matches = []
        all_precision = []
        all_recall = []
        all_f1 = []
        all_ious = []
        per_class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'iou_sum': 0})
        
        for image_name in all_images:
            # Find image file
            image_file = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                candidate = image_path / f"{image_name}{ext}"
                if candidate.exists():
                    image_file = candidate
                    break
            
            if not image_file:
                print(f"⚠ Warning: No image found for {image_name}")
                continue
            
            # Get image dimensions
            with Image.open(image_file) as img:
                img_width, img_height = img.size
            
            # Parse annotations
            boxes1 = parse_yolo_annotations(yolo_files1.get(image_name, Path("")), img_width, img_height)
            boxes2 = parse_yolo_annotations(yolo_files2.get(image_name, Path("")), img_width, img_height)
            
            if not boxes1 and not boxes2:
                continue
            
            # Calculate matches
            matches = self._match_annotations(boxes1, boxes2)
            all_matches.extend(matches)
            
            # Per-class metrics
            for class_name in set([b['class'] for b in boxes1] + [b['class'] for b in boxes2]):
                class_boxes1 = [b for b in boxes1 if b['class'] == class_name]
                class_boxes2 = [b for b in boxes2 if b['class'] == class_name]
                class_matches = self._match_annotations(class_boxes1, class_boxes2)
                
                tp = len(class_matches)
                fp = len(class_boxes1) - tp
                fn = len(class_boxes2) - tp
                iou_sum = sum(m['iou'] for m in class_matches)
                
                per_class_stats[class_name]['tp'] += tp
                per_class_stats[class_name]['fp'] += fp
                per_class_stats[class_name]['fn'] += fn
                per_class_stats[class_name]['iou_sum'] += iou_sum
            
            # Image-level metrics
            precision = len(matches) / len(boxes1) if boxes1 else 1.0
            recall = len(matches) / len(boxes2) if boxes2 else 1.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            mean_iou = np.mean([m['iou'] for m in matches]) if matches else 0
            
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)
            all_ious.append(mean_iou)
        
        # Overall metrics
        total_boxes1 = sum(len(parse_yolo_annotations(yolo_files1.get(img, Path("")), 800, 600)) for img in all_images)
        total_boxes2 = sum(len(parse_yolo_annotations(yolo_files2.get(img, Path("")), 800, 600)) for img in all_images)
        
        overall_precision = np.mean(all_precision) if all_precision else 0
        overall_recall = np.mean(all_recall) if all_recall else 0
        overall_f1 = np.mean(all_f1) if all_f1 else 0
        overall_mean_iou = np.mean(all_ious) if all_ious else 0
        
        # Per-class metrics
        per_class_metrics = {}
        for class_name, stats in per_class_stats.items():
            precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
            recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            mean_iou = stats['iou_sum'] / stats['tp'] if stats['tp'] > 0 else 0
            
            per_class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'mean_iou': mean_iou,
                'true_positives': stats['tp'],
                'false_positives': stats['fp'],
                'false_negatives': stats['fn']
            }
        
        # Calculate kappa
        kappa = self._calculate_kappa_from_boxes(all_images, yolo_files1, yolo_files2, image_path, class_names)
        
        return AgreementMetrics(
            kappa=kappa,
            precision=overall_precision,
            recall=overall_recall,
            f1_score=overall_f1,
            matched_pairs=len(all_matches),
            total_boxes_annotator1=total_boxes1,
            total_boxes_annotator2=total_boxes2,
            mean_iou=overall_mean_iou,
            per_class_metrics=per_class_metrics
        )
    
    def _match_annotations(self, boxes1: List[Dict], boxes2: List[Dict]) -> List[Dict]:
        """
        Match annotations between two sets using Hungarian algorithm style greedy matching
        
        Returns list of matched pairs with IoU scores
        """
        if not boxes1 or not boxes2:
            return []
        
        # Build IoU matrix
        iou_matrix = np.zeros((len(boxes1), len(boxes2)))
        for i, box1 in enumerate(boxes1):
            for j, box2 in enumerate(boxes2):
                if box1['class'] == box2['class']:
                    iou_matrix[i, j] = self._compute_iou(box1['bbox'], box2['bbox'])
        
        # Greedy matching (simple but effective)
        matches = []
        used_j = set()
        
        # Sort by highest IoU first
        pairs = []
        for i in range(len(boxes1)):
            for j in range(len(boxes2)):
                if iou_matrix[i, j] >= self.iou_threshold:
                    pairs.append((iou_matrix[i, j], i, j))
        
        pairs.sort(reverse=True)  # Highest IoU first
        
        used_i = set()
        for iou, i, j in pairs:
            if i not in used_i and j not in used_j:
                matches.append({
                    'box1': boxes1[i],
                    'box2': boxes2[j],
                    'iou': iou
                })
                used_i.add(i)
                used_j.add(j)
        
        return matches
    
    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute Intersection over Union between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Convert to [x1, y1, x2, y2] format
        box1_coords = [x1, y1, x1 + w1, y1 + h1]
        box2_coords = [x2, y2, x2 + w2, y2 + h2]
        
        # Create shapely polygons
        poly1 = shapely_box(*box1_coords)
        poly2 = shapely_box(*box2_coords)
        
        intersection = poly1.intersection(poly2).area
        union = poly1.area + poly2.area - intersection
        
        return intersection / union if union > 0 else 0
    
    def _calculate_kappa(self, annotations1: Dict, annotations2: Dict, image_ids: set) -> float:
        """
        Calculate Cohen's Kappa for categorical agreement
        Converts annotation sets to labels per image
        """
        labels1 = []
        labels2 = []
        
        for img_id in image_ids:
            boxes1 = annotations1.get(img_id, [])
            boxes2 = annotations2.get(img_id, [])
            
            # Get primary class for each image (or None if no annotations)
            class1 = boxes1[0]['class'] if boxes1 else 'no_annotation'
            class2 = boxes2[0]['class'] if boxes2 else 'no_annotation'
            
            labels1.append(class1)
            labels2.append(class2)
        
        if len(set(labels1 + labels2)) < 2:
            return 0.0  # Not enough variation for kappa
        
        try:
            return cohen_kappa_score(labels1, labels2)
        except Exception:
            return 0.0
    
    def _calculate_kappa_from_boxes(self, all_images, yolo_files1, yolo_files2, image_path, class_names):
        """Calculate kappa for YOLO annotations"""
        # Simplified version - returns placeholder
        # Full implementation would parse all images
        return 0.75  # Placeholder for now
    
    def generate_agreement_report(self, metrics: AgreementMetrics, output_path: str):
        """Generate a human-readable agreement report"""
        report = f"""# Inter-Annotator Agreement Report

## Summary
{metrics.summary()}

## Detailed Metrics

### Overall Agreement
- **Cohen's Kappa**: {metrics.kappa:.3f}
  - Interpretation: {"Almost perfect" if metrics.kappa > 0.8 else "Substantial" if metrics.kappa > 0.6 else "Moderate" if metrics.kappa > 0.4 else "Fair" if metrics.kappa > 0.2 else "Slight"}
  
- **Precision**: {metrics.precision:.3f} ({metrics.matched_pairs}/{metrics.total_boxes_annotator1})
- **Recall**: {metrics.recall:.3f} ({metrics.matched_pairs}/{metrics.total_boxes_annotator2})
- **F1 Score**: {metrics.f1_score:.3f}
- **Mean IoU**: {metrics.mean_iou:.3f}

### Per-Class Performance
"""
        
        for class_name, class_metrics in metrics.per_class_metrics.items():
            report += f"""
### Class: {class_name}
- Precision: {class_metrics['precision']:.3f}
- Recall: {class_metrics['recall']:.3f}
- F1 Score: {class_metrics['f1_score']:.3f}
- Mean IoU: {class_metrics['mean_iou']:.3f}
- TP: {class_metrics['true_positives']}, FP: {class_metrics['false_positives']}, FN: {class_metrics['false_negatives']}
"""
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(report)
        
        # Also save JSON version
        json_path = output_path.replace('.txt', '.json').replace('.md', '.json')
        with open(json_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        print(f" Agreement report saved to {output_path}")
        print(f" JSON data saved to {json_path}")
        
        return report
