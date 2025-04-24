#import torch
#import mlflow
#import os

# Set the tracking URI for your DagsHub repository
mlflow.set_tracking_uri("https://dagshub.com/samisena/MRI_VScode.mlflow")

# Set your DagsHub credentials
os.environ['MLFLOW_TRACKING_USERNAME'] = 'samisena'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'cb6584b07822e49dbde524fb92b0376756fcfdfd'

# Model configurations
MODEL_CONFIG = {
    'resnet50': {
        'run_id': "4a013ab305c843f1a5b073e13387c25f",
        'model_path': "final_model_resnet50",
    },
    'resnet100': {
        'run_id': '5bd6d7c47b154be388248ae12bd9c118',
        'model_path': "final_model_resnet100",
    },
    'efficientnetb0': {
        'run_id': '02f90bcb38e0410b9fa2490efb378d26',
        'model_path': "final_model_efficientnetb0",
    },
    'efficientnetb1': {
        'run_id': '821c6819b15949bf8a0eeb241310b89a',
        'model_path': "final_model_efficientnetb1",
    }
}

def find_target_layers(model, top_level_only=False):
    """
    Find potential target layers for GradCAM in a model.
    
    Args:
        model: The PyTorch model to inspect
        top_level_only: If True, only print top-level modules
    
    Returns:
        List of layer paths that could be used for GradCAM
    """
    # Dictionary to store model structure
    model_structure = {}
    target_layers = []
    
    # Function to recursively extract model structure
    def extract_structure(module, prefix=""):
        for name, child in module.named_children():
            current_path = f"{prefix}.{name}" if prefix else name
            
            # Check if it's a Conv2d layer
            if isinstance(child, torch.nn.Conv2d):
                info = f"Conv2d (in={child.in_channels}, out={child.out_channels}, kernel={child.kernel_size})"
                model_structure[current_path] = info
                target_layers.append(current_path)
            
            # If it's a container-like module, recursively process it
            if list(child.named_children()):
                extract_structure(child, current_path)
            # If it's a leaf module (not Conv2d)
            elif not isinstance(child, torch.nn.Conv2d):
                model_structure[current_path] = type(child).__name__
    
    # Start extraction from the root model
    extract_structure(model)
    
    if top_level_only:
        # Print only top-level modules
        top_modules = {}
        for path in model_structure:
            top_module = path.split('.')[0]
            if top_module not in top_modules:
                top_modules[top_module] = []
            top_modules[top_module].append(path)
        
        for module, paths in top_modules.items():
            print(f"{module}: {len(paths)} submodules")
    else:
        # Print all modules
        for path, info in model_structure.items():
            if "Conv2d" in info:
                print(f"{path}: {info}")
    
    return target_layers

def load_and_examine_model(model_name):
    """
    Load a model and examine its structure to find target layers
    
    Args:
        model_name: Name of the model to load
    """
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    
    # Get model configuration
    config = MODEL_CONFIG[model_name]
    run_id = config['run_id']
    model_uri = f"runs:/{run_id}/{config['model_path']}"
    
    try:
        # Load the model
        print(f"Loading model from MLflow...")
        model = mlflow.pytorch.load_model(model_uri, map_location=torch.device('cpu'))
        model.eval()
        
        # First, print top-level structure for an overview
        print("\nTop-level model structure:")
        print("-" * 40)
        for name, module in model.named_children():
            module_type = type(module).__name__
            num_parameters = sum(p.numel() for p in module.parameters())
            print(f"{name} ({module_type}): {num_parameters:,} parameters")
        
        # Find and print all Conv2d layers
        print("\nAll convolutional layers:")
        print("-" * 40)
        all_layers = find_target_layers(model)
        
        # Find last convolutional layer in each main block
        print("\nPotential target layers for GradCAM:")
        print("-" * 40)
        
        # Group layers by their first component
        layer_groups = {}
        for layer in all_layers:
            main_component = layer.split('.')[0]
            if main_component not in layer_groups:
                layer_groups[main_component] = []
            layer_groups[main_component].append(layer)
        
        # Print the last layer in each group
        for group, layers in layer_groups.items():
            if layers:
                last_layer = layers[-1]
                print(f"Last layer in {group}: {last_layer}")
        
        # Find the overall last convolutional layer
        if all_layers:
            last_conv = all_layers[-1]
            print(f"\nOverall last convolutional layer: {last_conv}")
        
    except Exception as e:
        print(f"Error loading or examining model: {e}")

def main():
    """Main function to examine all models"""
    for model_name in MODEL_CONFIG.keys():
        load_and_examine_model(model_name)

if __name__ == "__main__":
    main()