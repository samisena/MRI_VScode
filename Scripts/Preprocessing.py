import os
import sys
from pathlib import Path
from PIL import Image
from zipfile import ZipFile

def unzip_file(zip_path: str | Path, extract_path: str | Path) -> None: 
    """
    unzips a file to a given location

    Args:
        zip_path (Path): path to zip file
        extract_path (Path): directory where to extract the file to
    """

    try:
        zip_path = Path(zip_path)
        extract_path = Path(extract_path)
    except TypeError:
        raise TypeError("Both arguments must be strings or Path objects.")
     #? Convert the paths to Path objects to handle Windows paths properly
     #? This automatically handles backslashes and forward slashes appropriately  
     
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    #? The 'with' statement ensures that the ZIP file is properly closed 
    #? after we're done with it, even if an error occurs.
    #? zipfile.ZipFile() opens the ZIP file. The 'r' parameter means we're 
    #? opening it in read mode.


def verify_folders(testing_path: str | Path, training_path: str | Path) -> str:
    
    #? This function uses annotations:
    #? The -> syntax indicates the expected return type of the function.
    #? The : syntax indicates that the function accepts arguments with the specified types.
    
    """
    Verifies that both the testing and training folders exist.
    
    Args:
        testing_path (str | Path): Path to the testing folder
        training_path (str | Path): Path to the training folder
    
    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if both folders exist, False otherwise
            - str: Status message explaining the result
    """
    
    try:
        testing_path = Path(testing_path)
        training_path = Path(training_path)
    except TypeError:
        raise TypeError("Both arguments must be strings or Path objects.")

    # Check if both paths exist and are directories
    testing_exists = testing_path.exists() and testing_path.is_dir()
    training_exists = training_path.exists() and training_path.is_dir()
    
    # Create a detailed status message
    if testing_exists and training_exists:
        return True, "Both folders exist and are valid directories"
    
    # Construct error message for missing folders
    missing_folders = []    #Start by making an empty list
    
    if not testing_exists:
        missing_folders.append("testing")
        
    if not training_exists:
        missing_folders.append("training")
        
    error_message = f"Missing folders: {', '.join(missing_folders)}"   
    #? The .join() method is being called on the string ', ' (a comma followed by a space).
    #? This method takes an iterable (misssing_folders) and combines all its elements 
    #?  into a single string with elements seprated by ", " from each other
    return error_message

    # Prints: "Data/Raw/Testing"
    
    
#! Data set Class
#* 1. Correctly label each image data point
#* 2. Load each image from it's folder using Dataset Class
#* 3. Configure proper error handling and annotations

class MRIDataset:
    def __init__(self, root_dir: str | Path, split: str, transform:torchvision.transforms.Compose):
        """
        Args:
            data_dir (Path): Path to the directory where the data is located
            split (str): takes either "train" or "test"
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        #* Storing the initialisation parameters
        self.root_dir = Path(root_dir)
        self.split = 'training' if split == 'train' else 'testing'
        #! if split == 'train':
        #!     mode = 'training'
        #!  else:
        #!     mode = 'testing'
        self.transform = transform 
        
        #* Define class names and create class-to-index mapping
        self.classes = ['glioma', 'meningioma', 'pituitary', 'notumour'] 
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes)}
        #! for idx, cls in enumerate(self.classes):
        #! class_to_idx[cls] = idx
        
        #* Assigning each image to a class
        self.samples = []
        split_dir = self.root_dir / self.split  #? Creates Data/Raw/Testing or Data/Raw/Training
        

        
