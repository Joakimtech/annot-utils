from setuptools import setup, find_packages

setup(
    name="annot-utils",
    version="0.1.0",
    author="Your Name",
    description="Annotation format converter and quality checker",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "shapely>=2.0.0",
        "Pillow>=10.0.0",
        "click>=8.1.0",
        "scikit-learn>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "annot-utils=annot_utils.cli:main",
        ],
    },
    python_requires=">=3.8",
)
