"""
Simple tests to verify annot-utils works correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported"""
    try:
        from annot_utils.converter import AnnotationConverter
        from annot_utils.validator import AnnotationValidator
        from annot_utils.agreement import AgreementCalculator
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_version():
    """Test version attribute exists"""
    try:
        from annot_utils import __version__
        print(f"✅ Version: {__version__}")
        return True
    except ImportError:
        print("❌ Version not found")
        return False

def test_converter_creation():
    """Test converter can be instantiated"""
    try:
        from annot_utils.converter import AnnotationConverter
        converter = AnnotationConverter(verbose=False)
        print("✅ Converter created")
        return True
    except Exception as e:
        print(f"❌ Converter creation failed: {e}")
        return False

if __name__ == "__main__":
    tests = [
        test_imports,
        test_version,
        test_converter_creation,
    ]
    
    results = [test() for test in tests]
    
    if all(results):
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
