"""
Command-line interface for annotation utilities
Provides convert, validate, and agreement commands
"""

import click
import json
import sys
from pathlib import Path
from typing import Optional
import webbrowser

from .converter import AnnotationConverter
from .validator import AnnotationValidator
from .agreement import AgreementCalculator


@click.group()
@click.version_option(version="0.1.0")
def main():
    """
    Annotation Utilities - Convert, validate, and analyze annotation formats
    
    A comprehensive toolkit for computer vision annotation workflows.
    
    \b
    Examples:
        annot-utils convert --from coco --to yolo --input annotations.json --output ./yolo/
        annot-utils validate --annotations labels.json --format coco --images ./images/
        annot-utils agreement --annotator1 user1.json --annotator2 user2.json --format coco
    """
    pass


@main.command()
@click.option('--from', 'from_format', required=True, type=click.Choice(['coco', 'yolo']), 
              help='Source annotation format')
@click.option('--to', 'to_format', required=True, type=click.Choice(['coco', 'yolo']),
              help='Target annotation format')
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='Input file or directory')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output file or directory')
@click.option('--class-names', '-c', help='Class names file or comma-separated list (for YOLO→COCO)')
@click.option('--image-dir', '-d', type=click.Path(exists=True), 
              help='Image directory (for YOLO→COCO)')
@click.option('--quiet', '-q', is_flag=True, help='Suppress verbose output')
def convert(from_format, to_format, input, output, class_names, image_dir, quiet):
    """
    Convert annotations between COCO and YOLO formats
    
    \b
    Examples:
        # COCO to YOLO
        annot-utils convert --from coco --to yolo --input annotations.json --output ./yolo_labels/
        
        # YOLO to COCO
        annot-utils convert --from yolo --to coco --input ./yolo/ --output coco.json --image-dir ./images/ --class-names cat,dog,bird
    """
    click.echo(f"🔄 Converting {from_format} → {to_format}...")
    
    converter = AnnotationConverter(verbose=not quiet)
    
    try:
        if from_format == 'coco' and to_format == 'yolo':
            stats = converter.coco_to_yolo(input, output)
            
            if not quiet:
                click.echo(f"\n✅ Conversion complete!")
                click.echo(f"   📊 {stats['images_with_annotations']}/{stats['total_images']} images converted")
                click.echo(f"   📝 {stats['total_annotations']} total annotations")
                click.echo(f"   📁 Output: {output}")
                
        elif from_format == 'yolo' and to_format == 'coco':
            if not class_names:
                click.echo("❌ Error: --class-names required for YOLO → COCO conversion", err=True)
                sys.exit(1)
            
            if not image_dir:
                click.echo("❌ Error: --image-dir required for YOLO → COCO conversion", err=True)
                sys.exit(1)
            
            # Parse class names
            if Path(class_names).exists():
                with open(class_names, 'r') as f:
                    class_names_list = [line.strip() for line in f if line.strip()]
            else:
                class_names_list = [c.strip() for c in class_names.split(',')]
            
            stats = converter.yolo_to_coco(input, image_dir, output, class_names_list)
            
            if not quiet:
                click.echo(f"\n✅ Conversion complete!")
                click.echo(f"   📊 {stats['images_with_annotations']}/{stats['total_images']} images converted")
                click.echo(f"   📝 {stats['total_annotations']} total annotations")
                click.echo(f"   📁 Output: {output}")
        
        else:
            click.echo(f"❌ Unsupported conversion: {from_format} → {to_format}", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Conversion failed: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--annotations', '-a', required=True, type=click.Path(exists=True),
              help='Annotations file or directory')
@click.option('--format', '-f', 'anno_format', required=True, type=click.Choice(['coco', 'yolo']),
              help='Annotation format')
@click.option('--images', '-i', required=True, type=click.Path(exists=True),
              help='Images directory')
@click.option('--output', '-o', type=click.Path(), help='Output report file (JSON or HTML)')
@click.option('--html/--no-html', default=True, help='Generate HTML report (default: True)')
@click.option('--min-box-size', default=100, type=int, help='Minimum box area in pixels')
@click.option('--max-aspect-ratio', default=10.0, type=float, help='Maximum width/height ratio')
@click.option('--strict', is_flag=True, help='Exit with error code if issues found')
def validate(annotations, anno_format, images, output, html, min_box_size, max_aspect_ratio, strict):
    """
    Validate annotations for quality and integrity
    
    \b
    Examples:
        # Validate COCO annotations
        annot-utils validate --format coco --annotations labels.json --images ./images/
        
        # Validate YOLO annotations with strict mode
        annot-utils validate --format yolo --annotations ./yolo_labels/ --images ./images/ --strict
    """
    click.echo(f"🔍 Validating {anno_format} annotations...")
    
    validator = AnnotationValidator(
        min_box_size=min_box_size,
        max_aspect_ratio=max_aspect_ratio
    )
    
    try:
        if anno_format == 'coco':
            report = validator.validate_coco(annotations, images)
        else:  # yolo
            # Need class names for YOLO validation
            class_names_file = click.prompt("Class names file path (or 'skip' to skip class validation)", 
                                           default="skip", show_default=False)
            if class_names_file != 'skip' and Path(class_names_file).exists():
                with open(class_names_file, 'r') as f:
                    class_names = [line.strip() for line in f if line.strip()]
                report = validator.validate_yolo(annotations, images, class_names)
            else:
                # Validate without class checking
                class_names = ['class_0', 'class_1', 'class_2']  # Placeholder
                report = validator.validate_yolo(annotations, images, class_names)
        
        # Display summary
        click.echo(f"\n📊 Validation Summary:")
        click.echo(f"   Total Images: {report.total_images}")
        click.echo(f"   Total Annotations: {report.total_annotations}")
        click.echo(f"   Errors: {len(report.errors)} ❌")
        click.echo(f"   Warnings: {len(report.warnings)} ⚠️")
        click.echo(f"   Info: {len(report.infos)} ℹ️")
        click.echo(f"   Quality Score: {report.quality_score}/100")
        
        # Show sample issues
        if report.errors:
            click.echo(f"\n❌ Sample errors:")
            for error in report.errors[:3]:
                click.echo(f"   • {error.description}")
        
        if report.warnings:
            click.echo(f"\n⚠️ Sample warnings:")
            for warning in report.warnings[:3]:
                click.echo(f"   • {warning.description}")
        
        # Generate report
        if output:
            if html and anno_format == 'coco':
                validator.generate_html_report(report, output)
                click.echo(f"\n📄 HTML report saved to: {output}")
                
                # Option to open in browser
                if click.confirm("   Open report in browser?", default=False):
                    webbrowser.open(f"file://{Path(output).absolute()}")
            else:
                # Save JSON report
                json_output = output if output.endswith('.json') else f"{output}.json"
                with open(json_output, 'w') as f:
                    json.dump(report.to_dict(), f, indent=2)
                click.echo(f"\n📄 JSON report saved to: {json_output}")
        
        # Exit with error code if strict mode and issues found
        if strict and (report.errors or report.warnings):
            click.echo(f"\n❌ Validation failed with {len(report.errors)} errors and {len(report.warnings)} warnings", err=True)
            sys.exit(1)
        
        click.echo("\n✅ Validation complete!")
        
    except Exception as e:
        click.echo(f"❌ Validation failed: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--annotator1', '-1', required=True, type=click.Path(exists=True),
              help='First annotator file/directory')
@click.option('--annotator2', '-2', required=True, type=click.Path(exists=True),
              help='Second annotator file/directory')
@click.option('--format', '-f', 'anno_format', required=True, type=click.Choice(['coco', 'yolo']),
              help='Annotation format')
@click.option('--output', '-o', type=click.Path(), help='Output report file')
@click.option('--iou-threshold', default=0.5, type=float, help='IoU threshold for box matching')
@click.option('--image-dir', '-i', type=click.Path(exists=True), 
              help='Images directory (required for YOLO format)')
@click.option('--class-names', '-c', help='Class names file (required for YOLO format)')
def agreement(annotator1, annotator2, anno_format, output, iou_threshold, image_dir, class_names):
    """
    Calculate inter-annotator agreement metrics
    
    \b
    Examples:
        # COCO format
        annot-utils agreement --format coco --annotator1 user1.json --annotator2 user2.json
        
        # YOLO format
        annot-utils agreement --format yolo --annotator1 ./user1/ --annotator2 ./user2/ --image-dir ./images/ --class-names classes.txt
    """
    click.echo(f"📊 Calculating inter-annotator agreement ({anno_format})...")
    click.echo(f"   IoU Threshold: {iou_threshold}")
    
    calculator = AgreementCalculator(iou_threshold=iou_threshold)
    
    try:
        if anno_format == 'coco':
            metrics = calculator.calculate_coco_agreement(annotator1, annotator2)
            
        else:  # yolo
            if not image_dir:
                click.echo("❌ Error: --image-dir required for YOLO format", err=True)
                sys.exit(1)
            
            if not class_names:
                click.echo("❌ Error: --class-names required for YOLO format", err=True)
                sys.exit(1)
            
            # Load class names
            if Path(class_names).exists():
                with open(class_names, 'r') as f:
                    class_names_list = [line.strip() for line in f if line.strip()]
            else:
                class_names_list = [c.strip() for c in class_names.split(',')]
            
            metrics = calculator.calculate_yolo_agreement(annotator1, annotator2, image_dir, class_names_list)
        
        # Display results
        click.echo(metrics.summary())
        
        # Save report
        if output:
            calculator.generate_agreement_report(metrics, output)
            click.echo(f"\n📄 Full report saved to: {output}")
        
        # Provide interpretation
        click.echo("\n📈 Interpretation:")
        if metrics.f1_score >= 0.8:
            click.echo("   ✅ Excellent agreement - Annotators are highly consistent")
        elif metrics.f1_score >= 0.6:
            click.echo("   👍 Good agreement - Annotations are reliable")
        elif metrics.f1_score >= 0.4:
            click.echo("   ⚠️ Moderate agreement - Some inconsistencies need review")
        else:
            click.echo("   ❌ Poor agreement - Major discrepancies found")
        
        if metrics.kappa > 0.6:
            click.echo("   ✅ Strong categorical agreement beyond chance")
        elif metrics.kappa > 0.4:
            click.echo("   ⚠️ Moderate categorical agreement")
        else:
            click.echo("   ❌ Low categorical agreement - Review labeling guidelines")
        
    except Exception as e:
        click.echo(f"❌ Agreement calculation failed: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='Input annotation file')
@click.option('--format', '-f', 'anno_format', required=True, type=click.Choice(['coco', 'yolo']),
              help='Annotation format')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output statistics file (JSON)')
def stats(input, anno_format, output):
    """
    Generate detailed statistics about annotations
    
    \b
    Examples:
        annot-utils stats --format coco --input annotations.json --output stats.json
    """
    click.echo(f"📈 Generating statistics for {anno_format} annotations...")
    
    try:
        if anno_format == 'coco':
            with open(input, 'r') as f:
                data = json.load(f)
            
            stats_data = {
                'format': 'COCO',
                'total_images': len(data['images']),
                'total_annotations': len(data['annotations']),
                'total_categories': len(data['categories']),
                'categories': {},
                'bbox_statistics': {
                    'areas': [],
                    'aspect_ratios': []
                }
            }
            
            # Per-category counts
            for cat in data['categories']:
                stats_data['categories'][cat['name']] = 0
            
            for ann in data['annotations']:
                cat_name = next((c['name'] for c in data['categories'] if c['id'] == ann['category_id']), 'unknown')
                stats_data['categories'][cat_name] += 1
                
                # Bbox stats
                _, _, w, h = ann['bbox']
                area = w * h
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
                stats_data['bbox_statistics']['areas'].append(area)
                stats_data['bbox_statistics']['aspect_ratios'].append(aspect_ratio)
            
            # Compute statistics
            areas = stats_data['bbox_statistics']['areas']
            aspects = stats_data['bbox_statistics']['aspect_ratios']
            
            stats_data['bbox_statistics']['mean_area'] = float(np.mean(areas)) if areas else 0
            stats_data['bbox_statistics']['std_area'] = float(np.std(areas)) if areas else 0
            stats_data['bbox_statistics']['min_area'] = float(np.min(areas)) if areas else 0
            stats_data['bbox_statistics']['max_area'] = float(np.max(areas)) if areas else 0
            stats_data['bbox_statistics']['mean_aspect_ratio'] = float(np.mean(aspects)) if aspects else 0
            
            # Remove raw lists for cleaner output
            del stats_data['bbox_statistics']['areas']
            del stats_data['bbox_statistics']['aspect_ratios']
            
        else:  # yolo
            stats_data = {
                'format': 'YOLO',
                'total_images': len(list(Path(input).glob('*.txt'))),
                'total_annotations': 0,
                'classes': {}
            }
            
            for txt_file in Path(input).glob('*.txt'):
                with open(txt_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            class_name = f"class_{class_id}"
                            stats_data['classes'][class_name] = stats_data['classes'].get(class_name, 0) + 1
                            stats_data['total_annotations'] += 1
        
        # Save statistics
        with open(output, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        click.echo(f"\n✅ Statistics saved to: {output}")
        
        # Display summary
        click.echo(f"\n Summary:")
        click.echo(f"   Total Images: {stats_data['total_images']}")
        click.echo(f"   Total Annotations: {stats_data['total_annotations']}")
        
        if anno_format == 'coco':
            click.echo(f"   Total Categories: {stats_data['total_categories']}")
            click.echo(f"   Mean Box Area: {stats_data['bbox_statistics']['mean_area']:.0f} px²")
        
        click.echo(f"\n   Top classes:")
        sorted_classes = sorted(stats_data.get('categories', stats_data.get('classes', {})).items(), 
                               key=lambda x: x[1], reverse=True)[:5]
        for class_name, count in sorted_classes:
            click.echo(f"   • {class_name}: {count}")
        
    except Exception as e:
        click.echo(f"❌ Statistics generation failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
