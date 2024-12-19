"""
MedPathways Explorer - A LlamaIndex-powered medical school analysis tool
"""

from llama_index.core import Settings
from .explorer import Explorer
from .beagle import Beagle
from .utils import setup_logging

__version__ = "0.1.0"

# Default LlamaIndex configuration
Settings.chunk_size = 1024
Settings.chunk_overlap = 20
Settings.num_output = 512

__all__ = ["Explorer", "Beagle", "setup_logging"]
