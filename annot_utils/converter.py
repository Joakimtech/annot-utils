
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image


class AnnotationConverter:
    """Convert between different annotation formats"""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize converter
        
        Args:
            verbose: Print progress information
        """
        self.verbose = verbose
        
    def log(self, message: str, level: str = "INFO"):
        """Print log message if verbose is enabled"""
        if self.verbose:
            print(f"[{level}] {message}")
    
    def coco_to_yolo(self, coco_path: str, output_dir: str, use_relative_paths: bool = True):
        """
        Convert COCO JSON to YOLO format
        
        Args:
            coco_path: Path to COCO JSON file
            output_dir: Directory to save YOLO .txt files
            use_relative_paths: Use relative paths for output
            
        Returns:
            dict: Conversion statistics
        """
        self.log(f"Converting COCO → YOLO: {coco_path}")
        
        # Load COCO JSON
        try:
            with open(coco_path, 'r') as f:
                coco_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"COCO file not found: {coco_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON file: {coco_path}")
        
        # Build lookups
        images = {img['id']: img for img in coco_data['images']}
        categories = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}  # YOLO uses 0-indexed
        category_names = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Group annotations by image
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert each image
        stats = {
            'total_images': len(images),
            'images_with_annotations': 0,
            'total_annotations': 0,
            'converted_files': []
        }
        
        for img_id, img_info in images.items():
            img_filename = Path(img_info['file_name']).stem
            img_width = img_info['width']
            img_height = img_info['height']
            
            annotations = annotations_by_image.get(img_id, [])
            
            if annotations:
                stats['images_with_annotations'] += 1
                stats['total_annotations'] += len(annotations)
                
                # Create YOLO format file
                yolo_path = output_path / f"{img_filename}.txt"
                
                with open(yolo_path, 'w') as f:
                    for ann in annotations:
                        # COCO bbox: [x, y, width, height]
                        x, y, w, h = ann['bbox']
                        
                        # Convert to YOLO format: [center_x, center_y, width, height] normalized
                        center_x = (x + w / 2) / img_width
                        center_y = (y + h / 2) / img_height
                        norm_w = w / img_width
                        norm_h = h / img_height
                        
                        # Get class ID
                        category_id = ann['category_id']
                        class_id = categories.get(category_id)
                        
                        if class_id is None:
                            self.log(f"Warning: Unknown category ID {category_id} in image {img_filename}", "WARNING")
                            continue
                        
                        # Write YOLO line: class_id center_x center_y width height
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                
                stats['converted_files'].append(str(yolo_path))
                self.log(f"  ✓ Converted {img_filename}: {len(annotations)} annotations")
            else:
                self.log(f"  ○ Skipped {img_filename}: no annotations")
        
        # Save class mapping file
        class_map_path = output_path / "class_mapping.json"
        reverse_mapping = {v: k for k, v in categories.items()}
        class_names = {v: category_names[reverse_mapping[v]] for v in reverse_mapping}
        
        with open(class_map_path, 'w') as f:
            json.dump({
                'class_id_to_name': class_names,
                'class_name_to_id': {name: idx for idx, name in class_names.items()}
            }, f, indent=2)
        
        self.log(f" Conversion complete: {stats['images_with_annotations']}/{stats['total_images']} images with {stats['total_annotations']} total annotations")
        self.log(f" Output directory: {output_path}")
        self.log(f" Class mapping saved to: {class_map_path}")
        
        return stats
    
    def yolo_to_coco(self, yolo_dir: str, image_dir: str, output_json: str, 
                     class_names: Optional[List[str]] = None, class_mapping_file: Optional[str] = None):
        """
        Convert YOLO .txt files to COCO JSON format
        
        Args:
            yolo_dir: Directory containing YOLO .txt files
            image_dir: Directory containing corresponding images
            output_json: Path to output COCO JSON file
            class_names: List of class names (index = class ID)
            class_mapping_file: JSON file with class mapping (alternative to class_names)
            
        Returns:
            dict: Conversion statistics
        """
        self.log(f"Converting YOLO → COCO: {yolo_dir}")
        
        yolo_path = Path(yolo_dir)
        image_path = Path(image_dir)
        
        # Load class mapping
        if class_mapping_file:
            with open(class_mapping_file, 'r') as f:
                mapping_data = json.load(f)
                # Handle different mapping formats
                if 'class_id_to_name' in mapping_data:
                    class_names = [mapping_data['class_id_to_name'][str(i)] for i in range(len(mapping_data['class_id_to_name']))]
                elif 'class_name_to_id' in mapping_data:
                    class_names = list(mapping_data['class_name_to_id'].keys())
                else:
                    class_names = list(mapping_data.values()) if isinstance(mapping_data, dict) else mapping_data
        
        if not class_names:
            raise ValueError("Either class_names or class_mapping_file must be provided")
        
        # Prepare COCO structure
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': [
                {
                    'id': i + 1,  # COCO uses 1-indexed categories
                    'name': name,
                    'supercategory': 'none'
                }
                for i, name in enumerate(class_names)
            ]
        }
        
        # Find all YOLO files
        yolo_files = list(yolo_path.glob('*.txt'))
        
        if not yolo_files:
            raise ValueError(f"No .txt files found in {yolo_dir}")
        
        # Statistics
        stats = {
            'total_images': len(yolo_files),
            'images_with_annotations': 0,
            'total_annotations': 0,
            'converted_images': []
        }
        
        image_id = 1
        annotation_id = 1
        
        for yolo_file in sorted(yolo_files):
            # Find corresponding image
            image_name = yolo_file.stem
            image_file = None
            
            # Check for common image extensions
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                candidate = image_path / f"{image_name}{ext}"
                if candidate.exists():
                    image_file = candidate
                    break
            
            if not image_file:
                self.log(f"⚠ Warning: No image found for {yolo_file.name}", "WARNING")
                continue
            
            # Get image dimensions
            try:
                with Image.open(image_file) as img:
                    width, height = img.size
            except Exception as e:
                self.log(f"⚠ Warning: Cannot read image {image_file}: {e}", "WARNING")
                continue
            
            # Add image to COCO
            coco_data['images'].append({
                'id': image_id,
                'file_name': image_file.name,
                'width': width,
                'height': height
            })
            
            # Read YOLO annotations
            annotations = []
            try:
                with open(yolo_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if len(parts) != 5:
                            self.log(f"⚠ Warning: {yolo_file.name} line {line_num}: Expected 5 values, got {len(parts)}", "WARNING")
                            continue
                        
                        try:
                            class_id, cx, cy, w, h = map(float, parts)
                        except ValueError:
                            self.log(f"⚠ Warning: {yolo_file.name} line {line_num}: Invalid number format", "WARNING")
                            continue
                        
                        # Validate class ID
                        class_id_int = int(class_id)
                        if class_id_int < 0 or class_id_int >= len(class_names):
                            self.log(f"⚠ Warning: {yolo_file.name} line {line_num}: Class ID {class_id_int} out of range (0-{len(class_names)-1})", "WARNING")
                            continue
                        
                        # Convert YOLO normalized coordinates to COCO absolute
                        x = (cx - w/2) * width
                        y = (cy - h/2) * height
                        bbox_width = w * width
                        bbox_height = h * height
                        
                        # Validate bbox coordinates
                        if bbox_width <= 0 or bbox_height <= 0:
                            self.log(f"⚠ Warning: {yolo_file.name} line {line_num}: Invalid bbox dimensions", "WARNING")
                            continue
                        
                        annotation = {
                            'id': annotation_id,
                            'image_id': image_id,
                            'category_id': class_id_int + 1,  # COCO uses 1-indexed
                            'bbox': [x, y, bbox_width, bbox_height],
                            'area': bbox_width * bbox_height,
                            'iscrowd': 0,
                            'segmentation': []  # Empty for bounding boxes
                        }
                        coco_data['annotations'].append(annotation)
                        annotations.append(annotation)
                        annotation_id += 1
                        
            except Exception as e:
                self.log(f"⚠ Warning: Error reading {yolo_file.name}: {e}", "WARNING")
                continue
            
            if annotations:
                stats['images_with_annotations'] += 1
                stats['total_annotations'] += len(annotations)
                stats['converted_images'].append(str(image_file))
                self.log(f"  ✓ Converted {image_name}: {len(annotations)} annotations")
            else:
                self.log(f"  ○ Skipped {image_name}: no valid annotations")
            
            image_id += 1
        
        # Save COCO JSON
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        self.log(f" Conversion complete: {stats['images_with_annotations']}/{stats['total_images']} images with {stats['total_annotations']} total annotations")
        self.log(f" Output file: {output_path}")
        
        return stats
    
    def batch_convert(self, input_path: str, output_path: str, 
                     conversion_type: str, **kwargs):
        """
        Batch convert multiple files
        
        Args:
            input_path: Input file or directory
            output_path: Output file or directory
            conversion_type: 'coco2yolo' or 'yolo2coco'
            **kwargs: Additional arguments for specific converters
        """
        if conversion_type == 'coco2yolo':
            # Handle single file or directory of COCO files
            input_dir = Path(input_path)
            if input_dir.is_file():
                return self.coco_to_yolo(input_path, output_path, **kwargs)
            else:
                results = []
                for coco_file in input_dir.glob('*.json'):
                    result = self.coco_to_yolo(str(coco_file), output_path, **kwargs)
                    results.append(result)
                return results
                
        elif conversion_type == 'yolo2coco':
            return self.yolo_to_coco(input_path, output_path, **kwargs)
        else:
            raise ValueError(f"Unknown conversion type: {conversion_type}")
