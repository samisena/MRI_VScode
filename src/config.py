import os
from pathlib import Path

class Config:
    PROJECT_ROOT = Path(__file__).parent.parent   #* Goes to MRI_VSCODE workspace
    DATA_FOLDER = PROJECT_ROOT / 'Data'           
    TRAINING_DATA = DATA_FOLDER / 'Training'
    TESTING_DATA = DATA_FOLDER / 'Testing'