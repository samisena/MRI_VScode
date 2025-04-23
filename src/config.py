import os
from pathlib import Path

class Config:
    PROJECT_ROOT = Path(__file__).parent.parent   #* Goes to MRI_VSCODE workspace
    DATA_FOLDER = PROJECT_ROOT / 'Data'           
    TRAINING_DATA = DATA_FOLDER / 'Training'
    TESTING_DATA = DATA_FOLDER / 'Testing'
    MODEL_PATH = PROJECT_ROOT / 'mlruns' / '0' / '9dd4f79fb759487f8248a53ac261fe06' / 'artifacts' /'final_model_resnet50' / 'data' / 'model.pth'
    HTML_PATH = PROJECT_ROOT / 'Templates' / 'ui.html'
    HTML_MONITORING_PATH = PROJECT_ROOT / 'Templates' / 'monitoring.html'