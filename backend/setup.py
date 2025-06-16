"""Setup configuration for PHAL backend package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="phal",
    version="3.0.0",
    author="Jason DeLooze",
    author_email="jasonmarkd@gmail.com",
    description="Pluripotent Hardware Abstraction Layer for Controlled Environment Agriculture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HydroFarmerJason/PHAL",
    project_urls={
        "Bug Tracker": "https://github.com/HydroFarmerJason/PHAL/issues",
        "Documentation": "https://github.com/HydroFarmerJason/PHAL/tree/main/docs",
        "Source Code": "https://github.com/HydroFarmerJason/PHAL",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Home Automation",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "sqlalchemy>=2.0.0",
        "redis>=5.0.0",
        "influxdb-client>=1.38.0",
        "paho-mqtt>=1.6.0",
        "numpy>=1.26.0",
        "httpx>=0.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
        ],
        "hardware": [
            "pyserial>=3.5",
            "pymodbus>=3.5.0",
            "adafruit-circuitpython-ads1x15>=2.2.0",
        ],
        "ml": [
            "scikit-learn>=1.3.0",
            "pandas>=2.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "phal=phal.cli:main",
            "phal-api=phal.api:run",
        ],
    },
)
