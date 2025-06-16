"""
PHAL - Pluripotent Hardware Abstraction Layer

An open-source platform for controlled environment agriculture.
"""

__version__ = "3.0.0"
__author__ = "Jason DeLooze"
__license__ = "MIT"

from .core import Zone, Sensor, Actuator, Recipe
from .api import app

__all__ = ["Zone", "Sensor", "Actuator", "Recipe", "app", "__version__"]
