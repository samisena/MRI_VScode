from mri.Preprocessing import *
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt 
import mlflow
import mlflow.pytorch  #? MLflow's doesn't automatically import all submodules 

class Resnet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()                      #? we inhereit the instance attributes of nn.Module
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  #? we define the architecture
                                                     #? as being a that of a pre-trained resnet50 with default weights
        self.resnet.fc = nn.Linear(2048, num_classes) #? we change the last fully connected layer output neurons
                                                        #? to fit the number of classes
    def forward(self, x):
        return self.resnet(x)
    #* Note: it doesn't matter what how we define the instance attribute for the architecture
    #* as long as we call it in the forward method
        
<<<<<<< HEAD
class Resnet100(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet100(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(2048, num_classes)
    def forward(self, x):
        self.resnet(x)
=======
class Resnet101(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        #self.resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.resnet = models.resnet101(pretrained=True)


        self.resnet.fc = nn.Linear(2048, num_classes)
    def forward(self, x):
        return self.resnet(x)
>>>>>>> 3bb79b2d5b4d879f749828b42264c4d0fe23890d

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.efficientnet.classifier = nn.Linear(1280, num_classes)
    
    def forward(self,x):
        return self.efficientnet(x)
    
class EfficientNetB1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
<<<<<<< HEAD
        self.efficienet = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        self.efficient.classifier = nn.Linear(1280, num_classes)
    def forward(self, x):
        self.efficienet(x)
=======
        self.efficientnet = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        self.efficientnet.classifier = nn.Linear(1280, num_classes)
    def forward(self, x):
        return self.efficientnet(x)
>>>>>>> 3bb79b2d5b4d879f749828b42264c4d0fe23890d

def train_epoch(model, train_loader, criterion, optimizer, device) -> tuple:
    """
    This function defines how one epoch is trained
    
    args:
        model (torchvision.models | nn.Module): The model to train
        training_dataloader (torch.utils.data.DataLoader): the dataloader containing the training data
    """
    model.train()        #? Putting the model in train mode
    running_loss = 0.0   #? Resets the running_loss for each new epoch
    correct = 0
    total = 0
    
    for features, labels in train_loader:
        
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()      
        
        predictions = model(features)
        
        loss = criterion(predictions, labels)
        
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(predictions, 1)
        
        correct += (predicted==labels).sum().item()  #? counts the number of correct predictions
             
        total += labels.size(0)     #? returns the number of samples in each batch
               
    return running_loss/len(train_loader), (correct/total)*100   #! len(train_dataloader) returns the number
                                                        #! of batches not the number of samples in the dataset
        
        
def validate_epoch(model, val_loader, criterion, device) -> tuple:
    model.eval()    #? de-activates special layers like dropout
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():    #? stops pytorch from keeping computational graphs for each tensor (weight)
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            
            predictions = model(features)   #? outputs the predicted class index for each sample
            loss = criterion(predictions, labels)   #? outputs the loss given the correct classes
            
            val_loss += loss.item()
             
            _, predicted = torch.max(predictions, 1)   #? will return the index and the value of the 
                                                        #? largest value along dimension 1: the rows
                        
            correct += (predicted==labels).sum().item()  #? counts the number of correct predictions
             
            total += labels.size(0)     #? returns the number of samples in each batch
            
    return val_loss/len(val_loader), (correct/total)*100
        
#! Let's add checkpoints
#! training metric visualisations
#! and a load trained model function

def load_checkpoint(checkpoint_path: Path, model, optimizer, scheduler, device):
    
    """
    This function looks for a model training checkpoint file.

    Returns:
        start_epoch: the epoch that training should begin with 0 if no checkpoints were found
                    otherwise current epoch + 1
        best_model_state: the best weights so far in training
        best_accuracy: the best recoreded accuracy so far
        no_improvement_epochs = the number of consecutive epochs where the training accuracy hasn't increased
        history: a dictionary containing the savec history of the current training process
    """
    
    if checkpoint_path.exists():                                 #?  Path method
        print(f'Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=device)  #? to avoid errors if usisng a cpu
        
        model.load_state_dict(checkpoint['model_state'])  #! loads the tensor that maps neurons to their weights
                                                        #! we use load_state_dict() fot tensor dict that were
                                                        #! saved using torch.save()
        
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        start_epoch = checkpoint['epoch']+1  #* The new starting epoch
        best_model_state = checkpoint['model_state']
        best_accuracy = checkpoint['best_accuracy']
        no_improvement_epochs = checkpoint['no_improvement_epochs']
        history =  checkpoint['history']
        
        print(f'Resuming training from epoch {start_epoch} with best accuracy {best_accuracy}')
        return start_epoch, best_model_state, best_accuracy, no_improvement_epochs, history
    
    else:
        print('No checkpoint was found, starting training from epoch 0.')
        return 0, 0, 0,  {
            'train_loss': [], 'train_accuracy': [],
            'val_loss': [], 'val_accuracy': [],
            'learning_rates': []
        }

        
def get_save_path(model) -> Path:
    
    """_
    Creates a directory where the model parameter weights can be saved

    Returns:
        save_path: Path object of the directory where to save the model
    """
    
    model_path = Path(model.__class__.__name__)
    
    current_dir = Path(__file__)  #* will output: c:/Users/samis/OneDrive/Bureau/MRI_VScode/src/mri/Models.py
    
    project_root = current_dir.parent.parent.parent  #* c:/Users/samis/OneDrive/Bureau/MRI_VScode
    
    save_dir = project_root / 'Trained_models'
    
    save_dir.mkdir(parents = True, exist_ok=True)  #* Creates a new directory if doesn't already exist
    
    save_path = save_dir /f"{model_path}.pth" 
    
    return save_path
    #? c:/Users/samis/OneDrive/Bureau/MRI_VScode/Trained_model/Resnet50.pth
    

def plot_training_history(history: dict):
    """
    This function plots the training and validation accuracies accross the epochs of training.

    Args:
        history (dict): a dictionary containing the training history
    """
    
    plt.figure(figsize=(8,6))
    
    #* The number of epochs correspond to those in 'history' dictionary
    epochs = range(1, len(history['train_accuracy'] + 1))  
    
    plt.plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Progress')
    plt.legend
    
    plt.show()
    

def train_model(model, epochs: int, patience: int, learning_rate: float,  train_loader, val_loader, save_path: Path, 
                checkpoint_freq=5, resume_from_checkpoint: bool = False) -> tuple:
    """
    Given a model, this function trains it for a specific number of epochs, and it includes
    advanced techniques such as: 
        * Early stopping: stopping the training after if the validation accuracy stops improving after
                            a given number of epochs (patience)
        * Learning rate scheduling: automatically adjust the learning rate during training
        * Checkpoint saving
        
    Returns:
        best_model_state: a dictionary that maps each layer's name to its parameter values
        val_accuracy: the percentage of correctly classified instances
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    # Add model name as prefix for better organization
    checkpoint_path = save_path.parent / f"{model.__class__.__name__}_checkpoint.pt"

    # Initialize optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,    
        mode='min',   
        factor=0.1,   
        patience=3,   
        min_lr=1e-6   
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Fixed typo in 'exists()' method
    if resume_from_checkpoint and checkpoint_path.exists():    
        start_epoch, best_model_state, best_accuracy, no_improvement_epochs, history = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, device)  # Fixed typo in 'map_location'
    else:
        start_epoch = 0                
        best_accuracy = 0
        no_improvement_epochs = 0
        history = {                                    
            'train_loss': [], 'train_accuracy': [],
            'val_loss': [], 'val_accuracy': [],
            'learning_rates': []
        }   
    
    progress_bar = tqdm(range(start_epoch, epochs), desc='Training')
    try:  # Added try block for proper exception handling
        for epoch in progress_bar:
            train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy = validate_epoch(model, val_loader, criterion, device)
            
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            
            # Update training history - fixed typo in 'history'
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_accuracy)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            history['learning_rates'].append(current_lr)
            
            progress_bar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'train_accuracy': f'{train_accuracy:.2f}',
                'val_loss': f'{val_loss:.4f}',
                'val_accuracy': f'{val_accuracy:.2f}',
                'learning_rate': f'{current_lr:.6f}'
            })
        
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model_state = model.state_dict()
                torch.save(best_model_state, save_path)
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1   
                
            if no_improvement_epochs > patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            if (epoch + 1) % checkpoint_freq == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'best_accuracy': best_accuracy,
                    'no_improvement_epochs': no_improvement_epochs,
                    'history': history
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"\nCheckpoint saved at epoch {epoch+1}")
                
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving checkpoint...")
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_accuracy': best_accuracy,
            'no_improvement_epochs': no_improvement_epochs,
            'history': history
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")
        
    progress_bar.close()
                
    return best_model_state, best_accuracy
            
#! The best accuracy is the not necessarily the final accuracy due to the patience variable   
#! Need to add other metrics like F1 or recall       

#? We could also save the training history plot

<<<<<<< HEAD
def save_model_version(model, epochs, patience, checkpoint_freq, learning_rate, best_accuracy) -> None:
=======
def save_model_version(model, model_name, epochs, patience, checkpoint_freq, learning_rate, best_accuracy) -> None:
>>>>>>> 3bb79b2d5b4d879f749828b42264c4d0fe23890d
    """
    When called this function adds a log to the the MLflow database.
    
    Args:
        model: the model after it's been trained
        epochs: nbre of training epochs
        patience: patience nbre
        checkpoint_freq: interval of epochs between checkpoints
        learning_rate: initial learning rate
        best_accuracy: the model's highest achieved accuracy
        
    """
<<<<<<< HEAD
    
    with mlflow.start_run():
=======

    mlflow.set_experiment("1st experiment")
    mlflow.set_tracking_uri('https://dagshub.com/samisena/MRI_VScode.mlflow')
    
    with mlflow.start_run(run_name = model_name):
>>>>>>> 3bb79b2d5b4d879f749828b42264c4d0fe23890d
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('patience', patience)
        mlflow.log_param('checkpoint_freq', checkpoint_freq)
        mlflow.log_param("learning_rate", learning_rate)

        
        mlflow.log_metric('best_accuracy', best_accuracy)
        
<<<<<<< HEAD
        mlflow.pytorch.log_model(model, artifact_path='final_model_resnet50')
        
=======
        mlflow.pytorch.log_model(model, artifact_path=f'final_model_{model_name}')
        
def register_model(run_id: str, model_name: str):
    """
    This function registers a model on gatshub for later usage
    arguments:
        run_id
        model_name
    """
    model_uri= f"runs:/{run_id}/{model_name}"
    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"Registered model: {registered_model.name}, version: {registered_model.version}")
>>>>>>> 3bb79b2d5b4d879f749828b42264c4d0fe23890d

