from mri.Preprocessing import *
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights

class Resnet50(models.resnet50):
    def __init__(self, num_classes):
        super().__init__(ResNet50_Weights.DEFAULT)
        self.fc = nn.Linear(2048, num_classes)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#? One line conditional statement that sets device as the GPU if its available


def train_one_epoch(model, train_dataloader, criterion, optimizer, device) -> float:
    """
    This function defines how one epoch is trained
    
    args:
        model (torchvision.models | nn.Module): The model to train
        training_dataloader (torch.utils.data.DataLoader): the dataloader containing the training data
    
    """
    
    model.train()        #? Putting the model in train mode
    running_loss = 0.0   #? Resets the running_loss for each new epoch
    
    for features, labbels in train_dataloader:
        
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()      
        
        predictions = model(features)
        
        loss = criterion(predictions, labels)
        
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss/len(train_dataloader)
        
        
        






#* Model = resnet50
#* loss function = cross entropy (classification)
#* optimizer = Adam 
#* Other techniques: K-fold, learning rate scheduler, early stopping

#! Incremental saving 
#! Training: validation split, GPU not CPU
#! Evaluation metrics: confusion matrix, accuracy
#! Computational cost metrics: time complexity


