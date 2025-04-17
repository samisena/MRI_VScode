from mri.Model import *
import torch.nn.functional as F
import cv2
import numpy as np
import copy


def tensor_to_image(tensor, mean = [0.485, 0.456, 0.406],   
    std  = [0.229, 0.224, 0.225]):
    
    """
    Converts a pytorch tensor to a numpy image by reversing the tensor inital conversion operations
    
    args:
        tensor: the pytroch tensor representing an image's features
        mean, std: the mean normalisation values accross the 3 RGB channels
    """
    
    #? 1) Tensor to numpy array
    img = tensor.detach().cpu().squeeze(0).numpy() #shape -> [3, 224, 224]
    
    #? 2) Un-normalize via broadcasting
    mean = np.array(mean)[:, None, None]  #Broadcasts:  [3,] -> [3, 1, 1]
    std  = np.array(std)[:, None, None]
    img = img * std + mean    # un-normalizing ; [3, 224, 224]
    
    #? 3) Clip and convert to uint8
    img = np.clip(img, 0, 1)     #makes sure RGB values are between 0 and 1.
    img = (img * 255).astype(np.uint8)   #values are being 0 and 255 - and are 8 bit - image standard
    
    #? 4) Transpose 
    img = img.transpose(1, 2, 0)  #from [C, H, W] to [H, W, C]
    return img



def gradcam(model, target_layer, input_tensor, class_idx=None, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    This function computes gradcam visualizations.
    
    args:
        model: PyTorch model
        target_layer: The layer we'd like to extract feature maps and gradients from
        input_tensor: the image in tensor format
    
    """
    #! 1) Setting up the Forward and Backward hooks
    
    activations = []   # will hold the feature map values (forward pass) of the target layer
    gradients = []    # will hold the gradients of the target layer (backward pass)
    
    #? Functions to append data to the empty lists
    def forward_hook(module, input, output):
        """ A forward hook that PyTorch will call at every forward pass

        Args:
            module : the target layer. Usually the last one of convolution layer.
            input : the inputs that went into the module
            output : the output tensor produced by the module
        """
        activations.append(output.detach()) # Appends the moduel's outputs to activations
                                            # .detach() means PyTorch will not track operations for gradient
                                            # computation
                                            
    def backward_hook(module, grad_input, grad_output):
        """ A backward hook that PyTorch will call during backpropagation

        Args:
            module : the target layer. Usually the last one of convolution layer
            grad_input : gradients with respect to the module's inputs
            grad_output : gradient with respect to the module's outputs
        """
        gradients.append(grad_output[0].detach())  # output[0] corresponds to the output of the model
        
        
    if isinstance(target_layer, str):        # if the target layer variable is a string
        target_module = dict([*model.named_modules()])[target_layer]
    else:
        target_module = target_layer        # if not a string - use directly
        
    
    #? Attaches hooks to the target layer - passes 3 arguments to forward_hook and backward_hook 
    forward_handle = target_module.register_forward_hook(forward_hook) # .register_forward_hook()
                               # tells PyTorch to call forward_hook every time target_module runs a forward pass
    
    backward_handle = target_module.register_full_backward_hook(backward_hook)
    
    
    #! 2) Running Forward and backward pass
    
    model.eval()
    output = model(input_tensor)  #? Forward pass
    
    #? If no target class is provided, the code selects the class with the highest score (argmax)
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()  #.item() extracts a Python value '3' from tensor format '([3])'
        
    #? Clear previous gradients
    model.zero_grad()
    
    #? computes backpropagation gradients - backward hook captures the gradient values
    loss = output[0, class_idx]
    loss.backward()
    
    #? Removes the hooks - no longer needed
    forward_handle.remove()
    backward_handle.remove()
    
    #? Since batch size = 1, we take the only element of both lists
    grad = gradients[0]
    act = activations[0]
    
    #? Averagae the spatial dimensions of height and width to obtain one weight value per layer
    weights = grad.mean(dim=(2,3), keepdim=True)
    
    
    #! 3) Heatmap calculations
    grad = gradients[0]       # 1×C×H×W - C: number of feature map channels 
    act  = activations[0]      # 1×C×H×W
    
    #? We average the H and W to a single value to get a single weight per feature map channel:
    weights = grad.mean(dim=(2,3), keepdim=True)      # 1×C×1×1
    
    #?  Gradcam equation:
    cam     = F.relu((weights * act).sum(dim=1, keepdim=True))  # 1multiplies activations with their
                                     # weight scalar via broadcasting before passing the outputs
                                     # through a ReLU zeroing out negatives and making C 1 dimensional
                                     # 1xCxHxW -> 1x1xHxW  (2D image of weighted activation maps)
                                                                
    cam     = cam.squeeze().cpu()   #squeeze() removes dimensions of size 1: 1x1xHxW -> HxW
                                    #cpu() ensures compatibility with Numpy and cv2
                                    
    #? min-max normalization:
    heatmap = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # H×W float of [0,1]


    #! 4) Overlay heatmap over the original image
    
    #? Undoing tensor transformations:
    orig_img = tensor_to_image(input_tensor)

    h, w = orig_img.shape[:2]
    hm_uint8   = np.uint8(255 * heatmap)     # HxW values 0-255
    hm_resized = cv2.resize(hm_uint8, (w, h))  #match original size
    hm_inv     = 255 - hm_resized          # reverse grayscale
    
    hm_color_bgr = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)  # Turns grey heatmap to blue->red
    hm_color = hm_color_bgr[..., ::-1]   #RGB instead of default BGR
    overlay = cv2.addWeighted(orig_img, 1 - alpha, hm_color, alpha, 0)  # how strong the heatmap appears
                                                                        # over the original

    return heatmap, overlay



def guided_backprop(model, input_tensor, class_idx=None, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Performs Guided Backpropagation visualization on a PyTorch model with overlay option.
    
    Args:
        model: PyTorch model to visualize
        input_tensor: Input image tensor (1, C, H, W)
        class_idx: Target class index (uses predicted class if None)
        alpha: Transparency factor for the overlay (0.0 to 1.0)
        colormap: OpenCV colormap for visualization
    
    Returns:
        Tuple containing:
        - raw gradients visualization
        - overlay of gradients on original image
    """
    
    # Make a copy of the model - because we will alter the way ReLU layers work
    model_copy = copy.deepcopy(model)
    model_copy.eval()
    
    # Storage for forward activations (needed for guided backprop)
    activations = []
    
    # Define the modified ReLU class
    class GuidedReLU(torch.nn.Module):
        def forward(self, x):   # the forward method works as normal
            activations.append(x)  # input values are stored because we need to ensure 
             # we only calculate the gradients of positively activated neurons 
             # during backpropagation
        
            return torch.nn.functional.relu(x)
            
        def backward(self, grad_output):   # modified backpropagation
            x = activations.pop()   #.pop() ensures we get the last activations first
            positive_activation_mask = (x > 0).float()   # 1.0 if positive - 0 otherwise
            positive_grad_mask = (grad_output > 0).float()
            return positive_activation_mask * positive_grad_mask * grad_output 
            # This will zero out negative activations and gradients - and multiply
            # positive ones by 1.0 otherwise (filter)
    
    # Replacing all ReLU layers of our model with the modified GuidedReLU
    for name, module in model_copy.named_modules():  #loops through all modules of 
                                                     # the model   
        if isinstance(module, torch.nn.ReLU):   #checks if the current module has a
                                                #ReLU activation        
            # Extracts the parent module name:
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''  
            
            # Checks nested modules within parent modules
            if parent_name:
                parent = model_copy
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                    
                # Replace ReLU with our version
                layer_name = name.rsplit('.', 1)[1] if '.' in name else name
                setattr(parent, layer_name, GuidedReLU())  # line that replaces ReLUs
                                                           # with guided-ReLUs   
            else:
                # Top-level attribute
                setattr(model_copy, name, GuidedReLU())
    
    # Create a tensor that requires gradients for forward pass
    input_img = input_tensor.clone().detach().requires_grad_(True)
    
    # Forward pass
    output = model_copy(input_img)   # [batch_size, num_classes]
    
    # If no target class is provided, use the one with the highest score
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()
    
    # Zero all gradients of the model before backward pass
    model_copy.zero_grad()
    
    # Backward pass for target class
    one_hot = torch.zeros_like(output) # tensor of zeroes with the same shape as 'output'
    one_hot[0, class_idx] = 1  # puts a 1 at the target class position
    output.backward(gradient=one_hot) # backward pass only for the class with a '1' 
    
    # Get gradients from input
    gradients = input_img.grad.clone().detach()
    
    # Process gradients for visualization
    # Take absolute value of gradients
    grad_abs = np.abs(gradients.cpu().numpy()[0])
    
    # Normalize each channel separately for RGB visualization
    rgb_gradients = np.zeros_like(grad_abs)
    for i in range(3):
        channel_max = grad_abs[i].max() + 1e-8  # Avoid division by zero
        rgb_gradients[i] = grad_abs[i] / channel_max
    
    # Convert from (C,H,W) to (H,W,C) for visualization
    rgb_gradients = np.transpose(rgb_gradients, (1, 2, 0))
    
    # Create a grayscale representation for heatmap overlay
    grayscale_gradients = np.mean(grad_abs, axis=0)  # Average across channels
    
    # Normalize grayscale gradients to [0, 1]
    grayscale_norm = (grayscale_gradients - grayscale_gradients.min()) / (grayscale_gradients.max() - grayscale_gradients.min() + 1e-8)
    
    # Convert normalized gradients to heatmap
    heatmap = np.uint8(255 * grayscale_norm)
    
    # Get original image
    orig_img = tensor_to_image(input_tensor)
    
    # Resize heatmap to match original image dimensions
    h, w = orig_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Apply colormap to create colored heatmap
    heatmap_colored = cv2.applyColorMap(heatmap_resized, colormap)
    heatmap_colored = heatmap_colored[..., ::-1]  # Convert BGR to RGB
    
    # Create overlay
    overlay = cv2.addWeighted(orig_img, 1-alpha, heatmap_colored, alpha, 0)
    
    return rgb_gradients, overlay

def visualize_guided_backprop(model, test_loader, target_layer=None, alpha=0.5):
    """
    Visualize guided backpropagation for a model.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader containing test data
        target_layer: Not used for guided backprop but kept for API consistency
        alpha: Transparency for overlay
    
    Returns:
        Visualizations and overlay
    """
    # Get one batch from the test loader
    inputs, _ = next(iter(test_loader))
    
    # Select the first image in the batch
    input_tensor = inputs[0].unsqueeze(0)
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Run guided backpropagation
    gradients, overlay = guided_backprop(
        model=model,
        input_tensor=input_tensor,
        class_idx=None,  # Use predicted class
        alpha=alpha
    )
    
    # Display results
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(tensor_to_image(input_tensor))
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.title("Guided Backprop Gradients")
    plt.imshow(gradients)
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.title("Guided Backprop Overlay")
    plt.imshow(overlay)
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    return gradients, overlay