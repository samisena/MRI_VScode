import os
import sys
from pathlib import Path
from PIL import Image
from zipfile import ZipFile
import torch
from torch.utils.data import Dataset, DataLoader
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


class MRIDataset(Dataset):
    def __init__(self, dataset_dir: str | Path, transform: None | transforms.Compose = None):
        """
        Initialises instance attributes when a MRIDataset object is created.
        
        Args:
            data_dir (Path): Path to the directory where the data is located
            split (str): takes either "train" or "test"
            transform (callable, optional): Optional transform to be applied on a sample.
            
        Raises:
            FileNotFoundError: If the specified dataset directory does not exist, if one of the class subfolders is missing, 
            or if there are no images in the folder.
        """
        self.dataset_dir = Path(dataset_dir)
        
        #? Potential Erros Checks
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"The folder {dataset_dir} cannot be found at the given location.")
        
        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(f"The provided path {dataset_dir} is not a valid directory/folder.")
    
        #* Storing the initialisation parameters
      
        self.transform = transform 
        if transform == None:
            print("No data transformation inputed - No transformations will be applied. \n" 
                  "Please input either 'test_transforms' or 'train_transforms' as the 2nd argument")
        
        #* Defining the classes and numerically encoding them
        self.classes = ['glioma', 'meningioma', 'pituitary', 'notumor'] 
        #! Alternatively:  self.classes = [d.name for d in self.dataset_dir.iterdir()]
        
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes)}
        #! Dictionary Comprehension equivalent to:
        #! for idx, cls in enumerate(self.classes):
        #!      class_to_idx[cls] = idx
        
        #* Assigning each image to a numerical class
        self.samples = []                       
        for class_name in self.classes:
            class_dir = dataset_dir / class_name      #? Creates Data/Raw/Testing/glioma for example
            if not class_dir.exists():
                raise FileNotFoundError(f"The folder {class_dir} is missing.")
                
            class_idx = self.class_to_index[class_name]  #? [ ] because this is dictionary indexing
            
            for img_path in class_dir.glob('*.jpg'):
                self.samples.append((img_path, class_idx))
                    
            if not self.samples:    #! If self.sample is empty = False (List Comprehension)
                raise FileNotFoundError(f"No images found in {self.dataset_dir}")
                
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
 
 
testing_set_dir = Path.cwd().parent / "Data" / 'Raw' / "Testing"
training_set_dir = Path.cwd().parent / "Data" / 'Raw' / "Training"

train_dataset = MRIDataset(training_set_dir,  train_transforms)
test_dataset = MRIDataset(testing_set_dir,  test_transforms )

train_loader = DataLoader(train_dataset, batch_size=32,
                          shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32,
                         shuffle=True, num_workers=4)