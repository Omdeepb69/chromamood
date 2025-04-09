import setuptools
from pathlib import Path

_PROJECT_NAME = "ChromaMood"
_PROJECT_VERSION = "0.1.0"
_PROJECT_DESCRIPTION = "Analyzes the dominant colors in a live webcam feed or static image and suggests a mood or generates abstract art based on the color palette."
_PROJECT_AUTHOR = "Omdeep Borkar"
_PROJECT_AUTHOR_EMAIL = "omdeeborkar@gmail.com"
_PROJECT_URL = "https://github.com/Omdeepb69/ChromaMood"
_PROJECT_LICENSE = "MIT"
_PYTHON_REQUIRES = ">=3.7"

_INSTALL_REQUIRES = [
    "opencv-python",
    "numpy",
    "scikit-learn",
]

_THIS_DIR = Path(__file__).parent
try:
    _LONG_DESCRIPTION = (_THIS_DIR / "README.md").read_text()
except FileNotFoundError:
    _LONG_DESCRIPTION = _PROJECT_DESCRIPTION

_CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: End Users/Desktop",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Multimedia :: Graphics",
    "Topic :: Multimedia :: Video :: Capture",
    "Topic :: Scientific/Engineering :: Image Processing",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

setuptools.setup(
    name=_PROJECT_NAME,
    version=_PROJECT_VERSION,
    author=_PROJECT_AUTHOR,
    author_email=_PROJECT_AUTHOR_EMAIL,
    description=_PROJECT_DESCRIPTION,
    long_description=_LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=_PROJECT_URL,
    packages=setuptools.find_packages(),
    install_requires=_INSTALL_REQUIRES,
    classifiers=_CLASSIFIERS,
    python_requires=_PYTHON_REQUIRES,
    license=_PROJECT_LICENSE,
    keywords="color analysis mood webcam image processing computer vision abstract art",
)