import os
import sys
from pathlib import Path
from PIL import Image
from zipfile import ZipFile
import torch
from torch.utils.data import Dataset
from torchvision import transforms



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
    
    #* Construct error message for missing folders
    missing_folders = []    
    
    if not testing_exists:
        missing_folders.append("testing")
        
    if not training_exists:
        missing_folders.append("training")
        
    error_message = f"Missing folders: {', '.join(missing_folders)}"   
    return error_message     #? returns: Missing folders testing, training 


class MRIDataset(Dataset):
    def __init__(self, root_dir: str | Path, split: str, transform: None | transforms.Compose = None):
        """
        Initialises isntance attributes when a MRIDataset object is created.
        
        Args:
            data_dir (Path): Path to the directory where the data is located
            split (str): takes either "train" or "test"
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        #* Storing the initialisation parameters
        
        self.root_dir = Path(root_dir)
        
        self.split = 'training' if split == 'train' else 'testing'
        #! Conditional statement equivalent to:
        #! if split == 'train':
        #!     mode = 'training'
        #!  else:
        #!     mode = 'testing'
        
        self.transform = transform 
        
        #* Defining the classes and encoding them
        
        self.classes = ['glioma', 'meningioma', 'pituitary', 'notumour'] 
        
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes)}
        #! for idx, cls in enumerate(self.classes):
        #!      class_to_idx[cls] = idx
        
        #* Assigning each image to a class
        
        self.samples = []                       
        split_dir = self.root_dir / self.split      #? Creates Data/Raw/Testing or Data/Raw/Training
        for class_name in self.classes:
            class_dir = split_dir / class_name      #? Creates Data/Raw/Testing/glioma for example
            class_idx = self.class_to_index[class_name]  #? [ ] because this is dictionary indexing
            
            #* So far, we have got the class name and index of the current iterated folder
            #* Now we iterate over each image of the current class
            
            for img_path in class_dir.glob('*.{png, jpg, jpeg}'):
                self.samples.append((img_path, class_idx))
                
    def __getitem__(self, index):
        """
        A method used to retrieve and apply transformations to a data point given it's index.
        
        Returns:
            tuple: (image, class_idx)

        Args:
            index (_type_): _description_
        """
        
        img_path, class_idx = self.samples[index]
        
        image = Image.open(img_path)   #? The PIL library is compatible with Path() objects
        
        if self.transform is not None:
            image = self.transform(image) 
            #? Now it's a 224x224 PyTorch tensor, normalized and ready for the neural network

        return image, class_idx
    
    def __len__(self):
        """
        Returns:
            int: The number of samples in the dataset
        """
        return len(self.samples)
        

test_transforms = transforms.Compose([
    transforms.Resize((224,224)), 
    transforms.ToTensor(),
    transforms.Normalize(             
        mean = [0.485, 0.456, 0.406],   #* Image Net normalization
        std = [0.229, 0.224, 0.225]    
    )
])
        
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(10, fill=0),    #* 10 degrees - empty spaces are filled with black
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomHorizontalFlip(0.5),     #* 50% chances of a mirro flip (left side becomes right side)
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
    
])
 
#! Potential errors: Make sure all files are images - and make sure all images are either RGB or L
