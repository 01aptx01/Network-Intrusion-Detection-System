"""Main entry point for training network intrusion detection system model."""

import sys
from src.models.train_model import train_pipeline

if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, AttributeError):
                pass
    train_pipeline()
