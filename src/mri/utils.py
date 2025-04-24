import torch
import mlflow
import os
import sys
from config import Config       #* importing the config file
from pathlib import Path
from PIL import Image, ImageStat
from zipfile import ZipFile
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt 
import mlflow.pytorch  
import torch.nn.functional as F
import cv2
import numpy as np
import copy
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import io
import base64
from matplotlib.figure import Figure
from io import BytesIO
import time
import json
from datetime import datetime
from collections import defaultdict, deque
import threading
import logging
from fastapi import BackgroundTasks 
from dotenv import load_dotenv
