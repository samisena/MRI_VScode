

import os
import sys
print(f"Python version: {sys.version}")
print(f"Python location: {sys.executable}")
print(f"Working directory: {os.getcwd()}")

try:
    from PIL import Image
except ImportError as e:
    print(f"Detailed error: {e}")
    # On Windows, this will help show DLL loading issues
    import ctypes
    from ctypes import windll
    try:
        windll.LoadLibrary("_imaging")
    except Exception as dll_error:
        print(f"DLL error details: {dll_error}")