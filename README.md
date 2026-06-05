# annot-utils
A Python CLI tool for converting between annotation formats (COCO ↔ YOLO), validating label quality, and calculating inter-annotator agreement. Perfect for data annotation teams and computer vision workflows
## been vibe coding some of the files lately. More updates coming soon 

# 🎯 annot-utils

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

> A professional toolkit for computer vision annotation workflows - Convert, validate, and analyze annotations with ease.

**annot-utils** is a command-line tool that streamlines annotation management for computer vision projects. Whether you're working with COCO JSON or YOLO format, this tool helps you convert between formats, validate annotation quality, measure inter-annotator agreement, and generate detailed statistics.

## ✨ Features

| Feature | Description | Status |
|---------|-------------|--------|
| 🔄 **Format Conversion** | Bidirectional COCO ↔ YOLO conversion with batch processing | ✅ |
| ✓ **Validation** | Comprehensive annotation integrity checks with HTML reports | ✅ |
| 📊 **Agreement Analysis** | Cohen's Kappa, IoU matching, and per-class F1 scores | ✅ |
| 📈 **Statistics** | Detailed annotation statistics and distributions | ✅ |
| 🎨 **HTML Reports** | Beautiful, shareable validation reports | ✅ |
| 🚀 **Batch Processing** | Convert entire datasets with one command | ✅ |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/joakimtech/annot-utils.git
cd annot-utils

# Install in development mode
pip install -e .

# Or install with pip directly
pip install annot-utils
