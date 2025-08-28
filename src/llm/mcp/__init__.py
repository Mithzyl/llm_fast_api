import sys
import os

# Add the 'src' directory to the Python path to resolve module imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Set ROOT_DIR environment variable
if "ROOT_DIR" not in os.environ:
    os.environ["ROOT_DIR"] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
