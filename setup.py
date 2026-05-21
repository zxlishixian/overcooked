#!/usr/bin/env python
from setuptools import find_packages, setup

with open("README.md", "r", encoding="UTF8") as fh:
    long_description = fh.read()

setup(
    name="overcooked",
    version="1.0.0",
    description="Overcooked-AI environment with multi-agent convergence speed research framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Shixian Li",
    url="https://github.com/zxlishixian/overcooked",
    packages=find_packages(),
    package_data={
        "overcooked_ai_py": [
            "data/layouts/*.layout",
            "data/planners/*.py",
            "data/human_data/*.pickle",
            "data/graphics/*.png",
            "data/graphics/*.json",
            "data/fonts/*.ttf",
        ],
    },
    install_requires=[
        "dill",
        "numpy<2.0.0",
        "scipy",
        "tqdm",
        "gymnasium",
        "ipython",
        "pygame",
        "ipywidgets",
        "opencv-python",
        "torch>=1.10.0",
        "matplotlib",
    ],
    python_requires=">=3.8",
)
