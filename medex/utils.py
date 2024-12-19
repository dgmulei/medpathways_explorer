import logging
from datetime import datetime
import os

def setup_logging(school: str, component: str) -> logging.Logger:
    """Configure component logging"""
    os.makedirs(f"{school}/logs", exist_ok=True)
    
    logger = logging.getLogger(f"{school}_{component}")
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    fh = logging.FileHandler(
        f"{school}/logs/{component}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger