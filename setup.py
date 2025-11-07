"""
Neurotronic Phase Caster - Setup Configuration
THz Coherence Wearable for Active Consciousness Engineering
"""

from setuptools import setup, find_packages
import pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="neurotronic-phase-caster",
    version="0.1.0-alpha",
    description="THz Coherence Wearable for Active Consciousness Engineering",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/9x25dillon/neurotronic_phase_caster",
    author="Chris Sweigard",
    author_email="chris@example.com",  # Update with actual email
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords="neuroscience consciousness terahertz EEG mental-health AI medical-device",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0,<2.0.0",
        "matplotlib>=3.5.0,<4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "yhwh-demo=demos.yhwh_demo_interactive:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/9x25dillon/neurotronic_phase_caster/issues",
        "Source": "https://github.com/9x25dillon/neurotronic_phase_caster",
        "Documentation": "https://github.com/9x25dillon/neurotronic_phase_caster/tree/main/docs",
    },
)
