from mri.Preprocessing import *
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.optim import Adam

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
        


def train_one_epoch(model, train_loader, criterion, optimizer, device) -> float:
    """
    This function defines how one epoch is trained
    
    args:
        model (torchvision.models | nn.Module): The model to train
        training_dataloader (torch.utils.data.DataLoader): the dataloader containing the training data
    """
    
    model.train()        #? Putting the model in train mode
    running_loss = 0.0   #? Resets the running_loss for each new epoch
    
    for features, labels in train_loader:
        
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()      
        
        predictions = model(features)
        
        loss = criterion(predictions, labels)
        
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss/len(train_loader)   #! len(train_dataloader) returns the number of batches not the 
                                                #! number of samples in the dataset
        
        
        
def validate_epoch(model, val_loader, criterion, device):
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
            
            
            
if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #? One line conditional statement that sets device as the GPU if its available
    
    model = Resnet50(num_classes=4)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = Adam(model.parameters(), lr=0.001)
    
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"The trainin loss is: {train_loss}")
    
    val_loss, val_prct = validate_epoch(model, val_loader, criterion,device )
    print(f"""The validation loss is {round(val_loss,2)}, 
          and the percentage of correctly classified instances is: {round(val_prct,2)}%""")
    
    

#* Model = resnet50
#* loss function = cross entropy (classification)
#* optimizer = Adam 
#* Other techniques: K-fold, learning rate scheduler, early stopping

#! Incremental saving 
#! Training: validation split, GPU not CPU
#! Evaluation metrics: confusion matrix, accuracy
#! Computational cost metrics: time complexity


