import sys
import os

print("\n=== DETAILED ENVIRONMENT INFO ===")
print(f"1. Working Directory: {os.getcwd()}")
print(f"2. Python Executable: {sys.executable}")
print(f"3. Python Version: {sys.version}")

print("\n=== PYTHONPATH ===")
for path in sys.path:
    print(f"- {path}")

print("\n=== CONDA INFO ===")
if 'CONDA_PREFIX' in os.environ:
    print(f"CONDA_PREFIX: {os.environ['CONDA_PREFIX']}")
    print(f"CONDA_DEFAULT_ENV: {os.environ.get('CONDA_DEFAULT_ENV', 'Not Set')}")
else:
    print("No Conda environment detected in environment variables")

# Now try imports
print("\n=== IMPORT TESTS ===")
try:
    import numpy
    print(f"Numpy found at: {numpy.__file__}")
except ImportError as e:
    print(f"Numpy import error: {e}")

try:
    import matplotlib
    print(f"Matplotlib found at: {matplotlib.__file__}")
except ImportError as e:
    print(f"Matplotlib import error: {e}")

try:
    from PIL import Image
    print(f"PIL found at: {Image.__file__}")
except ImportError as e:
    print(f"PIL import error: {e}")