from mri.Model import *
import torch.nn.functional as F
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt

def tensor_to_image(tensor):
    """Convert a PyTorch tensor to a numpy image for visualization"""
    # Assuming tensor is in format [1, C, H, W] with values normalized
    img = tensor.clone().detach().cpu().numpy()[0]
    
    # Transpose from [C, H, W] to [H, W, C] for visualization
    img = np.transpose(img, (1, 2, 0))
    
    # For normalized images, we need to denormalize
    # This assumes standard normalization with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    
    # Clip values to [0, 1] range
    img = np.clip(img, 0, 1)
    
    # Convert to 0-255 range for visualization
    img = (img * 255).astype(np.uint8)
    
    return img


def guided_backprop(model, input_tensor, class_idx=None):
    """
    Performs Guided Backpropagation visualization on a PyTorch model.
    
    Args:
        model: PyTorch model to visualize
        input_tensor: Input image tensor (1, C, H, W)
        class_idx: Target class index (uses predicted class if None)
    
    Returns:
        RGB visualization of guided gradients
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
    # Take absolute value for gradient intensity
    grad_abs = np.abs(gradients.cpu().numpy()[0])
    
    # Normalize each channel separately for RGB visualization
    rgb_gradients = np.zeros_like(grad_abs)
    for i in range(3):
        channel_max = grad_abs[i].max() + 1e-8  # Avoid division by zero
        rgb_gradients[i] = grad_abs[i] / channel_max
    
    # Convert from (C,H,W) to (H,W,C) for visualization
    rgb_gradients = np.transpose(rgb_gradients, (1, 2, 0))
    
    return rgb_gradients


def gradcam(model, target_layer, input_tensor, class_idx=None, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    This function computes gradcam visualizations.
    
    Args:
        model: PyTorch model
        target_layer: The layer we'd like to extract feature maps and gradients from
        input_tensor: the image in tensor format
        class_idx: Target class index (uses predicted class if None)
        alpha: Transparency factor for overlay
        colormap: OpenCV colormap for visualization
    
    Returns:
        Dictionary containing heatmap, overlay, and original image
    """
    # 1) Setting up the Forward and Backward hooks
    
    activations = []   # will hold the feature map values (forward pass) of the target layer
    gradients = []    # will hold the gradients of the target layer (backward pass)
    
    # Functions to append data to the empty lists
    def forward_hook(module, input, output):
        """ A forward hook that PyTorch will call at every forward pass """
        activations.append(output.detach())
                                            
    def backward_hook(module, grad_input, grad_output):
        """ A backward hook that PyTorch will call during backpropagation """
        gradients.append(grad_output[0].detach())
        
    if isinstance(target_layer, str):        # if the target layer variable is a string
        target_module = dict([*model.named_modules()])[target_layer]
    else:
        target_module = target_layer        # if not a string - use directly
        
    # Attaches hooks to the target layer
    forward_handle = target_module.register_forward_hook(forward_hook)
    backward_handle = target_module.register_full_backward_hook(backward_hook)
    
    # 2) Running Forward and backward pass
    
    model.eval()
    output = model(input_tensor)  # Forward pass
    
    # If no target class is provided, select the class with the highest score
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()
        
    # Clear previous gradients
    model.zero_grad()
    
    # Compute backpropagation gradients
    loss = output[0, class_idx]
    loss.backward()
    
    # Removes the hooks - no longer needed
    forward_handle.remove()
    backward_handle.remove()
    
    # 3) Heatmap calculations
    grad = gradients[0]       # 1×C×H×W - C: number of feature map channels 
    act = activations[0]      # 1×C×H×W
    
    # Average the gradients across spatial dimensions to get one weight per feature map channel
    weights = grad.mean(dim=(2,3), keepdim=True)      # 1×C×1×1
    
    # GradCAM equation - weighted combination of activation maps followed by ReLU
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
                                                                
    cam = cam.squeeze().cpu().numpy()   # convert to numpy HxW
                                    
    # min-max normalization
    heatmap = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # H×W float of [0,1]

    # 4) Overlay heatmap over the original image
    
    # Undoing tensor transformations:
    orig_img = tensor_to_image(input_tensor)

    h, w = orig_img.shape[:2]
    hm_uint8 = np.uint8(255 * heatmap)     # HxW values 0-255
    hm_resized = cv2.resize(hm_uint8, (w, h))  # match original size
    
    hm_color_bgr = cv2.applyColorMap(hm_resized, colormap)  # Apply colormap
    hm_color = hm_color_bgr[..., ::-1]   # RGB instead of default BGR
    overlay = cv2.addWeighted(orig_img, 1 - alpha, hm_color, alpha, 0)  # Weighted overlay

    return {
        'heatmap': heatmap,
        'overlay': overlay,
        'original': orig_img
    }


def guided_gradcam(model, target_layer, input_tensor, class_idx=None, use_relu=True, alpha=0.4):
    """
    Creates Guided GradCAM by combining Guided Backpropagation with GradCAM.
    
    Args:
        model: PyTorch model
        target_layer: Layer for GradCAM (string or module)
        input_tensor: Input image tensor (1, C, H, W)
        class_idx: Target class index (uses predicted class if None)
        use_relu: Whether to apply ReLU to the final result
        alpha: Transparency for overlay if needed
    
    Returns:
        Dictionary with guided_gradcam, overlay on original image, and individual components
    """
    # Get GradCAM result
    gradcam_result = gradcam(model, target_layer, input_tensor, class_idx, alpha)
    heatmap = gradcam_result['heatmap']  # This is already normalized to [0,1]
    original_img = gradcam_result['original']
    
    # Get Guided Backpropagation result - this is already in RGB format (H,W,3)
    guided_bp = guided_backprop(model, input_tensor, class_idx)
    
    # Resize heatmap to match guided backprop dimensions if needed
    h, w = guided_bp.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Expand heatmap to 3 channels for element-wise multiplication
    heatmap_3ch = np.expand_dims(heatmap_resized, axis=2)  # Shape becomes (H, W, 1)
    heatmap_3ch = np.repeat(heatmap_3ch, 3, axis=2)        # Shape becomes (H, W, 3)
    
    # Element-wise multiplication of guided backprop with heatmap
    guided_gradcam = guided_bp * heatmap_3ch
    
    # Apply ReLU (set negative values to 0) if requested
    if use_relu:
        guided_gradcam = np.maximum(guided_gradcam, 0)
    
    # Normalize for visualization
    guided_gradcam = guided_gradcam / (guided_gradcam.max() + 1e-8)
    
    # Create colored version for overlay
    guided_gradcam_uint8 = np.uint8(255 * guided_gradcam)
    
    # Create overlay on original image
    overlay = cv2.addWeighted(original_img, 1-alpha, guided_gradcam_uint8, alpha, 0)
    
    return {
        'guided_gradcam': guided_gradcam,
        'overlay': overlay,
        'gradcam_heatmap': heatmap,
        'guided_backprop': guided_bp,
        'original': original_img
    }


def visualize_guided_gradcam(model, test_loader, target_layer, class_idx=None, alpha=0.7, colormap=cv2.COLORMAP_JET):
    """
    Visualize Guided GradCAM for a model with enhanced color visualization.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader containing test data
        target_layer: Layer to use for GradCAM
        class_idx: Target class index (uses predicted class if None)
        alpha: Transparency factor for overlay (higher = more visible gradients)
        colormap: OpenCV colormap for visualization (e.g., cv2.COLORMAP_JET, 
                 cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_HOT, cv2.COLORMAP_RAINBOW)
    
    Returns:
        Visualizations of original image, guided gradients, and overlay
    """
    # Get one batch from the test loader
    inputs, _ = next(iter(test_loader))
    
    # Select the first image in the batch
    input_tensor = inputs[0].unsqueeze(0)
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Run guided gradcam with enhanced coloring
    result = guided_gradcam(
        model=model,
        target_layer=target_layer,
        input_tensor=input_tensor,
        class_idx=class_idx,
        alpha=alpha,
        colormap=colormap
    )
    
    # Display only the three requested outputs with enhanced color visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(result['original'])
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.title("Guided Gradients (Enhanced)")
    # Use the enhanced colored version of gradients for better visibility
    plt.imshow(result['enhanced_gradcam'])
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(result['overlay'])
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    return result


def enhanced_guided_backprop(model, input_tensor, class_idx=None, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """
    Enhanced guided backpropagation with overlay on original image and colormap
    
    Args:
        model: PyTorch model
        input_tensor: Input image tensor (1, C, H, W)
        class_idx: Target class index (uses predicted class if None)
        alpha: Transparency factor for overlay
        colormap: OpenCV colormap for visualization
    
    Returns:
        Dictionary with guided_bp, overlay, and original image
    """
    # Get original guided backpropagation result
    guided_bp = guided_backprop(model, input_tensor, class_idx)
    
    # Get original image
    original_img = tensor_to_image(input_tensor)
    
    # Convert guided_bp to heat visualization (more visible)
    # Normalize to 0-1 range
    guided_bp_normalized = (guided_bp - guided_bp.min()) / (guided_bp.max() - guided_bp.min() + 1e-8)
    
    # Convert to uint8 for colormap application
    guided_bp_uint8 = np.uint8(255 * guided_bp_normalized)
    
    # Apply colormap to make it more visible
    guided_bp_colored = cv2.applyColorMap(guided_bp_uint8, colormap)
    guided_bp_colored = cv2.cvtColor(guided_bp_colored, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Create overlay
    overlay = cv2.addWeighted(original_img, 1-alpha, guided_bp_colored, alpha, 0)
    
    return {
        'guided_bp': guided_bp,
        'guided_bp_colored': guided_bp_colored,
        'overlay': overlay,
        'original': original_img
    }


# Enhanced version of guided_gradcam that makes gradients more colorful and visible
def enhanced_guided_gradcam(model, target_layer, input_tensor, class_idx=None, alpha=0.7, colormap=cv2.COLORMAP_JET, 
                           brightness_factor=2.0, contrast_factor=1.5):
    """
    Enhanced guided GradCAM with more colorful and visible gradients
    
    Args:
        model: PyTorch model
        target_layer: Layer to use for GradCAM
        input_tensor: Input image tensor (1, C, H, W)
        class_idx: Target class index (uses predicted class if None)
        alpha: Transparency factor for overlay
        colormap: OpenCV colormap for visualization
        brightness_factor: Factor to increase brightness of visualization
        contrast_factor: Factor to increase contrast of visualization
    
    Returns:
        Dictionary with enhanced guided gradcam and original results
    """
    # Get original guided gradcam result
    original_result = guided_gradcam(model, target_layer, input_tensor, class_idx)
    
    # Get the guided gradcam visualization
    guided_gc = original_result['guided_gradcam']
    
    # Enhance brightness and contrast
    enhanced = np.clip(guided_gc * brightness_factor, 0, 1)  # Increase brightness
    enhanced = np.clip((enhanced - 0.5) * contrast_factor + 0.5, 0, 1)  # Increase contrast
    
    # Convert to uint8 for colormap application
    enhanced_uint8 = np.uint8(255 * enhanced)
    
    # Apply a more vibrant colormap
    enhanced_colored = cv2.applyColorMap(enhanced_uint8, colormap)
    enhanced_colored = cv2.cvtColor(enhanced_colored, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Create an enhanced overlay with higher alpha for more visibility
    original_img = original_result['original']
    enhanced_overlay = cv2.addWeighted(original_img, 1-alpha, enhanced_colored, alpha, 0)
    
    # Add results to original dictionary
    original_result['enhanced_gradcam'] = enhanced
    original_result['enhanced_colored'] = enhanced_colored
    original_result['enhanced_overlay'] = enhanced_overlay
    
    return original_result


def visualize_xai(model, input_tensor, target_layer):
    
    original_img = tensor_to_image(input_tensor)

    # 2. Enhanced Guided Backpropagation with overlay
    guided_bp_result = enhanced_guided_backprop(model, input_tensor, alpha=0.6)

    # 3. GradCAM
    gradcam_result = gradcam(model, target_layer, input_tensor)

    # 4. Enhanced Guided GradCAM with more vibrant colors
    guided_gradcam_result = enhanced_guided_gradcam(
        model, target_layer, input_tensor,  
        alpha=0.7, colormap=cv2.COLORMAP_JET, 
        brightness_factor=2.5, contrast_factor=2.0
    )

    # Create a figure with all visualizations
    plt.figure(figsize=(20, 15))

    # Original image
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_img)
    plt.axis("off")

    # Enhanced Guided Backpropagation (with overlay)
    plt.subplot(2, 2, 2)
    plt.title("Guided Backpropagation")
    plt.imshow(guided_bp_result['overlay'])
    plt.axis("off")

    # GradCAM
    plt.subplot(2, 2, 3)
    plt.title("GradCAM")
    plt.imshow(gradcam_result['overlay'])
    plt.axis("off")

    # Enhanced Guided GradCAM
    plt.subplot(2, 2, 4)
    plt.title("Enhanced Guided GradCAM")
    plt.imshow(guided_gradcam_result['enhanced_overlay'])
    plt.axis("off")

    plt.tight_layout()
    plt.show()
        