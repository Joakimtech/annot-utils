"""
Annotation validator - check annotation quality and integrity
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from PIL import Image
import numpy as np
from shapely.geometry import box as shapely_box


@dataclass
class ValidationIssue:
    """Represents a single validation issue"""
    issue_type: str
    severity: str  # 'error', 'warning', 'info'
    description: str
    location: Optional[Dict[str, Any]] = None
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ValidationReport:
    """Complete validation report for annotations"""
    total_images: int
    total_annotations: int
    errors: List[ValidationIssue]
    warnings: List[ValidationIssue]
    infos: List[ValidationIssue]
    statistics: Dict[str, Any]
    
    def to_dict(self):
        return {
            'total_images': self.total_images,
            'total_annotations': self.total_annotations,
            'errors': [e.to_dict() for e in self.errors],
            'warnings': [w.to_dict() for w in self.warnings],
            'infos': [i.to_dict() for i in self.infos],
            'statistics': self.statistics
        }
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def quality_score(self) -> float:
        """Calculate overall quality score (0-100)"""
        total_issues = len(self.errors) * 10 + len(self.warnings) * 2 + len(self.infos) * 0.5
        max_possible = self.total_annotations * 10 if self.total_annotations > 0 else 1
        score = max(0, 100 - (total_issues / max_possible * 100))
        return round(score, 2)


class AnnotationValidator:
    """Validate annotation quality and integrity"""
    
    def __init__(self, 
                 min_box_size: int = 100,  # Minimum area in pixels
                 max_box_size_ratio: float = 0.95,  # Max box size relative to image
                 max_aspect_ratio: float = 10.0,  # Max width/height ratio
                 min_confidence: float = 0.0):  # Minimum confidence score
        """
        Initialize validator with configurable thresholds
        
        Args:
            min_box_size: Minimum allowed box area in pixels
            max_box_size_ratio: Maximum box area as fraction of image area
            max_aspect_ratio: Maximum width/height ratio allowed
            min_confidence: Minimum confidence score (if available)
        """
        self.min_box_size = min_box_size
        self.max_box_size_ratio = max_box_size_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_confidence = min_confidence
    
    def validate_coco(self, coco_path: str, image_dir: str) -> ValidationReport:
        """
        Validate COCO format annotations
        
        Args:
            coco_path: Path to COCO JSON file
            image_dir: Directory containing images
            
        Returns:
            ValidationReport with all issues found
        """
        # Load COCO file
        with open(coco_path, 'r') as f:
            coco_data = json.load(f)
        
        # Build lookup dictionaries
        images = {img['id']: img for img in coco_data['images']}
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Group annotations by image
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # Initialize report
        report = ValidationReport(
            total_images=len(images),
            total_annotations=len(coco_data['annotations']),
            errors=[],
            warnings=[],
            infos=[],
            statistics={}
        )
        
        # Validate each image
        box_sizes = []
        aspect_ratios = []
        
        for img_id, img_info in images.items():
            img_path = self._find_image(image_dir, img_info['file_name'])
            
            # Get actual image dimensions if file exists
            if img_path and img_path.exists():
                with Image.open(img_path) as img:
                    actual_width, actual_height = img.size
                    if actual_width != img_info['width'] or actual_height != img_info['height']:
                        report.warnings.append(ValidationIssue(
                            issue_type="dimension_mismatch",
                            severity="warning",
                            description=f"Image dimensions mismatch: declared ({img_info['width']}x{img_info['height']}) vs actual ({actual_width}x{actual_height})",
                            location={'image_id': img_id, 'file': img_info['file_name']}
                        ))
                        img_width, img_height = actual_width, actual_height
                    else:
                        img_width, img_height = img_info['width'], img_info['height']
            else:
                img_width, img_height = img_info['width'], img_info['height']
                report.warnings.append(ValidationIssue(
                    issue_type="image_not_found",
                    severity="warning",
                    description=f"Image file not found: {img_info['file_name']}",
                    location={'image_id': img_id, 'file': img_info['file_name']}
                ))
            
            annotations = annotations_by_image.get(img_id, [])
            
            # Check for missing annotations
            if len(annotations) == 0:
                report.warnings.append(ValidationIssue(
                    issue_type="no_annotations",
                    severity="warning",
                    description=f"Image has no annotations",
                    location={'image_id': img_id, 'file': img_info['file_name']}
                ))
            
            # Validate each annotation
            for ann in annotations:
                self._validate_bbox(
                    ann['bbox'], 
                    img_width, 
                    img_height,
                    ann,
                    report,
                    img_info
                )
                
                # Collect statistics
                x, y, w, h = ann['bbox']
                area = w * h
                box_sizes.append(area)
                aspect_ratios.append(max(w, h) / min(w, h) if min(w, h) > 0 else float('inf'))
                
                # Validate category
                if ann['category_id'] not in categories:
                    report.errors.append(ValidationIssue(
                        issue_type="invalid_category",
                        severity="error",
                        description=f"Unknown category ID: {ann['category_id']}",
                        location={'image_id': img_id, 'annotation_id': ann['id'], 'file': img_info['file_name']}
                    ))
        
        # Check for overlapping boxes within same image
        for img_id, annotations in annotations_by_image.items():
            self._check_overlaps(annotations, report, images.get(img_id, {}))
        
        # Add statistics
        report.statistics = {
            'box_sizes': {
                'mean': float(np.mean(box_sizes)) if box_sizes else 0,
                'std': float(np.std(box_sizes)) if box_sizes else 0,
                'min': float(np.min(box_sizes)) if box_sizes else 0,
                'max': float(np.max(box_sizes)) if box_sizes else 0
            },
            'aspect_ratios': {
                'mean': float(np.mean(aspect_ratios)) if aspect_ratios else 0,
                'max': float(np.max(aspect_ratios)) if aspect_ratios else 0
            },
            'quality_score': report.quality_score
        }
        
        return report
    
    def validate_yolo(self, yolo_dir: str, image_dir: str, 
                     class_names: List[str]) -> ValidationReport:
        """
        Validate YOLO format annotations
        
        Args:
            yolo_dir: Directory containing YOLO .txt files
            image_dir: Directory containing images
            class_names: List of valid class names
            
        Returns:
            ValidationReport with all issues found
        """
        yolo_path = Path(yolo_dir)
        image_path = Path(image_dir)
        yolo_files = list(yolo_path.glob('*.txt'))
        
        report = ValidationReport(
            total_images=len(yolo_files),
            total_annotations=0,
            errors=[],
            warnings=[],
            infos=[],
            statistics={}
        )
        
        box_sizes = []
        aspect_ratios = []
        
        for yolo_file in yolo_files:
            # Find corresponding image
            image_file = self._find_image(image_path, yolo_file.stem)
            
            if not image_file or not image_file.exists():
                report.errors.append(ValidationIssue(
                    issue_type="image_not_found",
                    severity="error",
                    description=f"No image found for {yolo_file.name}",
                    location={'yolo_file': str(yolo_file)}
                ))
                continue
            
            # Get image dimensions
            with Image.open(image_file) as img:
                img_width, img_height = img.size
            
            # Read YOLO annotations
            annotations = []
            with open(yolo_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        report.errors.append(ValidationIssue(
                            issue_type="invalid_format",
                            severity="error",
                            description=f"Expected 5 values, got {len(parts)}",
                            location={'file': yolo_file.name, 'line': line_num}
                        ))
                        continue
                    
                    try:
                        class_id, cx, cy, w, h = map(float, parts)
                    except ValueError:
                        report.errors.append(ValidationIssue(
                            issue_type="invalid_number",
                            severity="error",
                            description=f"Invalid number format",
                            location={'file': yolo_file.name, 'line': line_num}
                        ))
                        continue
                    
                    # Validate class ID
                    if int(class_id) < 0 or int(class_id) >= len(class_names):
                        report.errors.append(ValidationIssue(
                            issue_type="invalid_class",
                            severity="error",
                            description=f"Class ID {int(class_id)} out of range (0-{len(class_names)-1})",
                            location={'file': yolo_file.name, 'line': line_num}
                        ))
                    
                    # Convert to absolute coordinates for validation
                    x = (cx - w/2) * img_width
                    y = (cy - h/2) * img_height
                    bbox_width = w * img_width
                    bbox_height = h * img_height
                    
                    annotation = {
                        'bbox': [x, y, bbox_width, bbox_height],
                        'class_id': int(class_id),
                        'confidence': None  # YOLO doesn't always have confidence
                    }
                    annotations.append(annotation)
                    
                    self._validate_bbox(
                        [x, y, bbox_width, bbox_height],
                        img_width,
                        img_height,
                        annotation,
                        report,
                        {'file_name': image_file.name}
                    )
                    
                    # Collect statistics
                    area = bbox_width * bbox_height
                    box_sizes.append(area)
                    aspect_ratios.append(max(bbox_width, bbox_height) / min(bbox_width, bbox_height))
            
            report.total_annotations += len(annotations)
            
            # Check for overlaps
            self._check_overlaps(annotations, report, {'file_name': image_file.name})
        
        # Add statistics
        report.statistics = {
            'box_sizes': {
                'mean': float(np.mean(box_sizes)) if box_sizes else 0,
                'std': float(np.std(box_sizes)) if box_sizes else 0,
                'min': float(np.min(box_sizes)) if box_sizes else 0,
                'max': float(np.max(box_sizes)) if box_sizes else 0
            },
            'aspect_ratios': {
                'mean': float(np.mean(aspect_ratios)) if aspect_ratios else 0,
                'max': float(np.max(aspect_ratios)) if aspect_ratios else 0
            },
            'quality_score': report.quality_score
        }
        
        return report
    
    def _validate_bbox(self, bbox: List[float], img_width: int, img_height: int,
                      annotation: Dict, report: ValidationReport, img_info: Dict):
        """Validate a single bounding box"""
        x, y, w, h = bbox
        
        # Check for negative values
        if x < 0 or y < 0 or w < 0 or h < 0:
            report.errors.append(ValidationIssue(
                issue_type="negative_coordinates",
                severity="error",
                description=f"Negative coordinates or dimensions: x={x}, y={y}, w={w}, h={h}",
                location={'image': img_info.get('file_name', 'unknown'), 'annotation': annotation.get('id', 'unknown')}
            ))
        
        # Check if box is within image boundaries
        if x + w > img_width or y + h > img_height:
            report.errors.append(ValidationIssue(
                issue_type="out_of_bounds",
                severity="error",
                description=f"Box extends beyond image ({img_width}x{img_height})",
                location={'image': img_info.get('file_name', 'unknown'), 'bbox': bbox}
            ))
        
        # Check minimum size
        area = w * h
        if area < self.min_box_size:
            report.warnings.append(ValidationIssue(
                issue_type="box_too_small",
                severity="warning",
                description=f"Box area {area:.0f}px < minimum {self.min_box_size}px",
                location={'image': img_info.get('file_name', 'unknown'), 'area': area}
            ))
        
        # Check maximum size
        max_area = img_width * img_height * self.max_box_size_ratio
        if area > max_area:
            report.warnings.append(ValidationIssue(
                issue_type="box_too_large",
                severity="warning",
                description=f"Box area {area:.0f}px exceeds {self.max_box_size_ratio*100:.0f}% of image",
                location={'image': img_info.get('file_name', 'unknown'), 'area': area}
            ))
        
        # Check aspect ratio
        if min(w, h) > 0:
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > self.max_aspect_ratio:
                report.warnings.append(ValidationIssue(
                    issue_type="extreme_aspect_ratio",
                    severity="warning",
                    description=f"Aspect ratio {aspect_ratio:.1f}:1 exceeds limit {self.max_aspect_ratio}:1",
                    location={'image': img_info.get('file_name', 'unknown'), 'aspect_ratio': aspect_ratio}
                ))
    
    def _check_overlaps(self, annotations: List[Dict], report: ValidationReport, img_info: Dict):
        """Check for overlapping bounding boxes"""
        if len(annotations) < 2:
            return
        
        boxes = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            boxes.append({
                'polygon': shapely_box(x, y, x + w, y + h),
                'annotation': ann
            })
        
        # Check each pair for significant overlap
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                intersection = boxes[i]['polygon'].intersection(boxes[j]['polygon']).area
                union = boxes[i]['polygon'].union(boxes[j]['polygon']).area
                
                if union > 0:
                    iou = intersection / union
                    if iou > 0.5:  # More than 50% overlap
                        report.infos.append(ValidationIssue(
                            issue_type="significant_overlap",
                            severity="info",
                            description=f"Boxes have {iou:.1%} overlap (IoU > 0.5)",
                            location={'image': img_info.get('file_name', 'unknown'), 'iou': iou}
                        ))
    
    def _find_image(self, image_dir: Path, filename: str) -> Optional[Path]:
        """Find image file with various extensions"""
        image_dir = Path(image_dir)
        
        # Try exact match first
        if (image_dir / filename).exists():
            return image_dir / filename
        
        # Try without extension (for YOLO files)
        stem = Path(filename).stem
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG']:
            candidate = image_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        
        return None
    
    def generate_html_report(self, report: ValidationReport, output_path: str):
        """Generate an HTML visualization of the validation report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Annotation Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .score {{ font-size: 48px; font-weight: bold; text-align: center; padding: 20px; }}
                .score-good {{ color: #4CAF50; }}
                .score-warning {{ color: #FF9800; }}
                .score-bad {{ color: #f44336; }}
                .summary {{ display: flex; gap: 20px; margin-bottom: 20px; }}
                .summary-card {{ flex: 1; background: #f9f9f9; padding: 15px; border-radius: 8px; text-align: center; }}
                .error {{ color: #f44336; border-left: 4px solid #f44336; padding: 10px; margin: 10px 0; background: #ffebee; }}
                .warning {{ color: #FF9800; border-left: 4px solid #FF9800; padding: 10px; margin: 10px 0; background: #fff3e0; }}
                .info {{ color: #2196F3; border-left: 4px solid #2196F3; padding: 10px; margin: 10px 0; background: #e3f2fd; }}
                h2 {{ margin-top: 20px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #667eea; color: white; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1> Annotation Validation Report</h1>
                    <p>Generated by annot-utils validator</p>
                </div>
                
                <div class="summary">
                    <div class="summary-card">
                        <h3>Total Images</h3>
                        <div style="font-size: 32px;">{report.total_images}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Total Annotations</h3>
                        <div style="font-size: 32px;">{report.total_annotations}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Quality Score</h3>
                        <div class="score {'score-good' if report.quality_score >= 80 else 'score-warning' if report.quality_score >= 60 else 'score-bad'}">
                            {report.quality_score}/100
                        </div>
                    </div>
                </div>
                
                <h2>✅ Issues Summary</h2>
                <div class="summary">
                    <div class="summary-card" style="background: #ffebee;">
                        <h3>Errors</h3>
                        <div style="font-size: 24px; color: #f44336;">{len(report.errors)}</div>
                    </div>
                    <div class="summary-card" style="background: #fff3e0;">
                        <h3>Warnings</h3>
                        <div style="font-size: 24px; color: #FF9800;">{len(report.warnings)}</div>
                    </div>
                    <div class="summary-card" style="background: #e3f2fd;">
                        <h3>Info</h3>
                        <div style="font-size: 24px; color: #2196F3;">{len(report.infos)}</div>
                    </div>
                </div>
                
                <h2>📈 Statistics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Mean Box Size</td><td>{report.statistics['box_sizes']['mean']:.0f} px²</td></tr>
                    <tr><td>Std Box Size</td><td>{report.statistics['box_sizes']['std']:.0f} px²</td></tr>
                    <tr><td>Min/Max Box Size</td><td>{report.statistics['box_sizes']['min']:.0f} / {report.statistics['box_sizes']['max']:.0f} px²</td></tr>
                    <tr><td>Mean Aspect Ratio</td><td>{report.statistics['aspect_ratios']['mean']:.2f}:1</td></tr>
                    <tr><td>Max Aspect Ratio</td><td>{report.statistics['aspect_ratios']['max']:.2f}:1</td></tr>
                </table>
                
                <h2> Errors ({len(report.errors)})</h2>
                {''.join([f'<div class="error"><strong>{e.issue_type}</strong>: {e.description}</div>' for e in report.errors])}
                
                <h2> Warnings ({len(report.warnings)})</h2>
                {''.join([f'<div class="warning"><strong>{w.issue_type}</strong>: {w.description}</div>' for w in report.warnings])}
                
                <h2> Info ({len(report.infos)})</h2>
                {''.join([f'<div class="info"><strong>{i.issue_type}</strong>: {i.description}</div>' for i in report.infos])}
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        print(f"✅ HTML report saved to {output_path}")
