from mri.Preprocessing import *
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.optim import Adam
from tqdm import tqdm

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
        

def train_epoch(model, train_loader, criterion, optimizer, device) -> float:
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
            
            
if __name__ == "__main__":
    model = Resnet50(num_classes=4)
    device = torch.device("cuda" if torch.cuda.is_available() == True else "cpu")
    model.to(device)   
     
    criterion = nn.CrossEntropyLoss()
    
    optimizer = Adam(model.parameters(), lr=0.001)
    
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"The trainin loss is: {train_loss}")
    
    val_loss, val_prct = validate_epoch(model, val_loader, criterion,device )
    print(f"""The validation loss is {round(val_loss,2)}, 
          and the percentage of correctly classified instances is: {round(val_prct,2)}%""")


def train_model(model, epochs, patience, train_loader, val_loader) -> tuple:
    """
    This function a given model for a specific number of epochs, and it includes
    advanced techniques such as: 
        * Early stopping: stopping the training after if the validation accuracy stops improving after
                            a given number of epochs (patience)
        * Learning rate scheduling: automatically adjust the learning rate during training
        * Hyperparameter tuning: learning rate, layer freezing & unfreezing
        * Incremental saving
        
    returns:
        best_model_state: a dictionary that maps each layer's name to its parameter values
        val_accuracy: the percentage of correctly classified instances
    """
    
    #* Adam optimizer and Learning rate scheduling
    optimizer = Adam(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,    #? where the learning rate to optimize is
        mode='min',   #? minimizing the validation loss
        factor = 0.1, #? multiply the learning rate by this factor when reducing
        patience = 3, #? number of epochs to wait before reducing the learning rate
        verbose = True, #? print statement when lr gets reduces
        min_lr=1e-6   #? minimum learning rate value
    )
    
    criterion = nn.CrossEntropyLoss()
    
    #? Metric that will keep track of the best model:
    best_accuracy = 0   
    
    
    #? Incremental saving
    history = {
        'train_loss':[],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rates': []
    }        
    
    #* Iterates over the number of epochs using tqdm progress bar:
    progress_bar = tqdm(range(epochs), desc='Training')
    for epoch in progress_bar:
        
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer)
        
        val_loss, val_accuracy = validate_epoch(model, val_loader, criterion )
        
        current_lr = optimizer.param_groups[0]['lr']  #?gets the current lr
        
        #! updates the lr if patience criteria is met:
        scheduler.step(val_loss)    #? direclty modifies the learning rate of optimizer
        
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['learning_rates'].append(current_lr)
        
        #* A progress bar instead of print statements:
        progress_bar.set_postfix({
            'train_loss': f'{train_loss:.4f}'
            'train_accuracy': f'{train_accuracy:.2f}'
            'val_loss': f'{val_loss:.4f}'
            'val_accuracy': f'{val_accuracy:.2f}'
            'learning_rate': f'{current_lr:.6f}'
        })
    
        
        #* We check if the current weights are the best so far, and if so we save them:
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy  #?we update the best accuracy
            best_model_state = model.state_dict()   #?saves the current parameters in a dictionary
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1   
            
        #* Early stopping:
        if no_improvement_epochs > patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
    progress_bar.close()
            
    return best_model_state, best_accuracy
            

if __name__ == "__main__":
    model = Resnet50(num_classes = 4)
    device = torch.device("cuda" if torch.cuda.is_available() == True else "cpu")
    model.to(device)
    train_model
    
    best_model_state, best_accuracy = train_model(model=model, epochs=10,
                                                  train_loader = train_loader,
                                                  val_loader=val_loader,
                                                  device=device)
    
    print(f"End of model training. Final accuracy: {best_accuracy:.2f}%")
    
    

    

        
        
        

