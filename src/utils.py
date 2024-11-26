import os

def verify_paths(paths):
    """
    Ensures that all required paths exist.
    """
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
