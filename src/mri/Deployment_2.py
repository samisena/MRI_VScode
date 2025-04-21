from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import io
import base64
from matplotlib.figure import Figure
from io import BytesIO
from config import Config
import mlflow
import os

# Import your existing model and XAI functions
from mri.ExplainableAI import *      

# Set the tracking URI for your DagsHub repository
mlflow.set_tracking_uri("https://dagshub.com/samisena/MRI_VScode.mlflow")

# Set your DagsHub credentials
os.environ['MLFLOW_TRACKING_USERNAME'] = 'samisena'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'cb6584b07822e49dbde524fb92b0376756fcfdfd'

# Model mapping dictionary with the CORRECT target layers
MODEL_CONFIG = {
    'resnet50': {
        'run_id': "4a013ab305c843f1a5b073e13387c25f",
        'model_path': "final_model_resnet50",
        'target_layer': "resnet.layer4.2.conv3"  # The last convolutional layer in ResNet50
    },
    'resnet100': {
        'run_id': '5bd6d7c47b154be388248ae12bd9c118',
        'model_path': "final_model_resnet100",
        'target_layer': "resnet.layer4.2.conv3"  # The last convolutional layer in ResNet100
    },
    'efficientnetb0': {
        'run_id': '02f90bcb38e0410b9fa2490efb378d26',
        'model_path': "final_model_efficientnetb0",
        'target_layer': "efficientnet.features.8.0"  # The last convolutional layer in EfficientNetB0
    },
    'efficientnetb1': {
        'run_id': '821c6819b15949bf8a0eeb241310b89a',
        'model_path': "final_model_efficientnetb1",
        'target_layer': "efficientnet.features.8.0"  # The last convolutional layer in EfficientNetB1
    }
}

# Class mapping for diagnosis results
CLASS_MAPPING = {
    3: "No Tumor (Healthy)", 
    0: "Glioma Tumor",       
    1: "Meningioma Tumor",  
    2: "Pituitary Tumor"    
}

# Dictionary to store loaded models (to avoid reloading the same model multiple times)
loaded_models = {}

# Function to load a model by name
def load_model(model_name):
    # If the model is already loaded, return it from cache
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    # Check if the model name is valid
    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Invalid model name: {model_name}")
    
    # Get model configuration
    config = MODEL_CONFIG[model_name]
    run_id = config['run_id']
    model_uri = f"runs:/{run_id}/{config['model_path']}"
    
    # Load the model
    model = mlflow.pytorch.load_model(model_uri, map_location=torch.device('cpu'))
    model.eval()
    print(f"Model {model_name} loaded successfully!")
    
    # Cache the model
    loaded_models[model_name] = model
    
    return model

# Create the FastAPI app
app = FastAPI()

# Set up static files directory for serving images
static_dir = Config.PROJECT_ROOT / 'static'
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Set up templates
templates = Jinja2Templates(directory=str(Config.PROJECT_ROOT / 'Templates'))

# Function to create web-friendly visualizations using your existing XAI functions
def visualize_xai_web(model, input_tensor, target_layer):
    """
    Creates visualizations for the web using the existing XAI functions.
    Returns a base64 encoded image containing all visualizations.
    
    Parameters:
    - model: The loaded model for which to generate visualizations
    - input_tensor: The preprocessed input tensor
    - target_layer: The specific target layer to use for GradCAM (depends on model architecture)
    """
    # Get the original image
    original_img = tensor_to_image(input_tensor)
    
    # Use your existing XAI functions
    guided_bp_result = enhanced_guided_backprop(model, input_tensor, alpha=0.6)
    gradcam_result = gradcam(model, target_layer, input_tensor)
    guided_gradcam_result = enhanced_guided_gradcam(
        model, target_layer, input_tensor,  
        alpha=0.7, colormap=cv2.COLORMAP_JET, 
        brightness_factor=2.5, contrast_factor=2.0
    )
    
    # Create a figure with all visualizations
    fig = Figure(figsize=(12, 10))
    
    # Original image
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("Original Brain Scan")
    ax1.imshow(original_img)
    ax1.axis("off")
    
    # Enhanced Guided Backpropagation
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("Guided Backpropagation")
    ax2.imshow(guided_bp_result['overlay'])
    ax2.axis("off")
    
    # GradCAM
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("GradCAM Heatmap")
    ax3.imshow(gradcam_result['overlay'])
    ax3.axis("off")
    
    # Enhanced Guided GradCAM
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title("Enhanced Guided GradCAM")
    ax4.imshow(guided_gradcam_result['enhanced_overlay'])
    ax4.axis("off")
    
    fig.tight_layout()
    
    # Save the figure to a BytesIO object
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    
    # Encode the image to base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    # Return the base64 encoded image
    return img_str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Root endpoint that serves the main UI page"""
    # Pass the list of available models to the template
    return templates.TemplateResponse("ui.html", {"request": request, "models": list(MODEL_CONFIG.keys())})

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_name: str = Form("resnet50")):
    """
    Process the uploaded image and return both prediction and visualization
    
    Parameters:
    - file: The uploaded MRI scan image
    - model_name: The name of the model to use for prediction (defaults to resnet50)
    """
    
    try:
        # Validate model name
        if model_name not in MODEL_CONFIG:
            return JSONResponse(
                content={"error": f"Invalid model name: {model_name}. Available models: {', '.join(MODEL_CONFIG.keys())}"},
                status_code=400
            )
        
        # Load the requested model
        model = load_model(model_name)
        
        # Get the appropriate target layer for this model
        target_layer = MODEL_CONFIG[model_name]['target_layer']
        
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        return JSONResponse(content={"error": f"Error processing request: {str(e)}"}, status_code=400)
    
    # Process the image for model input
    processed_image = test_transforms(image)
    input_tensor = processed_image.unsqueeze(0)  # Add batch dimension
    
    # Get model prediction
    with torch.no_grad():
        output = model(input_tensor)
        
    # Get the predicted class index and probability
    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    prediction_idx = output.argmax(dim=1).item()
    prediction_prob = probabilities[prediction_idx].item() * 100
    
    # Get the class name
    prediction_name = CLASS_MAPPING[prediction_idx]
    
    # Generate visualizations using your existing functions with model-specific target layer
    visualization_data = visualize_xai_web(model, input_tensor, target_layer=target_layer)
    
    # Return the prediction and visualization data
    return {
        "prediction": prediction_idx,
        "prediction_name": prediction_name,
        "confidence": f"{prediction_prob:.2f}%",
        "model_used": model_name,
        "visualization": visualization_data
    }
