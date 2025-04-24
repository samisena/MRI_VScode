from mri.ExplainableAI import *     

# Load environment variables from .env file if it exists
load_dotenv()
 

# Set the tracking URI for your DagsHub repository
mlflow.set_tracking_uri("https://dagshub.com/samisena/MRI_VScode.mlflow")

# Set your DagsHub credentials
#os.environ['MLFLOW_TRACKING_USERNAME'] = 'samisena'
#os.environ['MLFLOW_TRACKING_PASSWORD'] = 'd95798f0a5c4723a8ec0e4c2ed3fdeed5755e9a8'


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


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_monitoring.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mri_classifier_monitoring")

class ModelMonitor:
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.metrics = {
            "latency": defaultdict(lambda: deque(maxlen=window_size)),
            "confidence": defaultdict(lambda: deque(maxlen=window_size)),
            "predictions": defaultdict(lambda: deque(maxlen=window_size)),
            "image_stats": defaultdict(lambda: deque(maxlen=window_size)),
            "errors": defaultdict(lambda: deque(maxlen=window_size)),
        }
        self.class_counts = defaultdict(lambda: {cls: 0 for cls in CLASS_MAPPING})
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.lock = threading.Lock()

        # Create monitoring directory if it doesn't exist
        self.monitoring_dir = Config.PROJECT_ROOT / 'monitoring'
        self.monitoring_dir.mkdir(exist_ok=True)

        # Initialize monitoring data file if it doesn't exist
        self.monitoring_data_file = self.monitoring_dir / 'monitoring_data.json'
        if not self.monitoring_data_file.exists():
            with open(self.monitoring_data_file, 'w') as f:
                json.dump({}, f)
        
        # Load cumulative metrics from MLflow
        logger.info("Loading cumulative metrics from MLflow...")
        self.load_metrics_from_mlflow()
        
        logger.info(f"Model monitor initialized with data: {self.request_counts} requests, {self.error_counts} errors")

    def record_prediction(self, model_name, image_tensor, prediction_idx, confidence, latency, error=None):
        with self.lock:
            timestamp = datetime.now().isoformat()
            
            # Record metrics locally
            self.metrics["latency"][model_name].append(latency)
            self.metrics["confidence"][model_name].append(confidence)
            self.metrics["predictions"][model_name].append(prediction_idx)
            
            # Analyze image statistics if image is provided
            if image_tensor is not None:
                image_stats = self._compute_image_stats(image_tensor)
                self.metrics["image_stats"][model_name].append(image_stats)
            
            # Update class distribution counts and error tracking
            if error is None:
                # Class distribution update
                self.class_counts[model_name][prediction_idx] += 1
                self.request_counts[model_name] += 1
            else:
                # Error tracking
                self.metrics["errors"][model_name].append({
                    "error": str(error), 
                    "timestamp": timestamp
                })
                self.error_counts[model_name] += 1
            
            # Log the prediction - handle the case when prediction_idx is -1
            if prediction_idx in CLASS_MAPPING:
                prediction_name = CLASS_MAPPING[prediction_idx]
                logger.info(
                    f"Model: {model_name}, Prediction: {prediction_name}, "
                    f"Confidence: {confidence:.2f}%, Latency: {latency:.2f}ms"
                )
            else:
                # For error cases or unknown prediction indices
                logger.info(
                    f"Model: {model_name}, Prediction: Unknown (index: {prediction_idx}), "
                    f"Confidence: {confidence:.2f}%, Latency: {latency:.2f}ms"
                )
            
            # Calculate average metrics
            avg_latency = sum(self.metrics["latency"][model_name]) / max(len(self.metrics["latency"][model_name]), 1)
            avg_confidence = sum(self.metrics["confidence"][model_name]) / max(len(self.metrics["confidence"][model_name]), 1)
            
            # Calculate error rate
            total_requests = self.request_counts[model_name]
            error_count = self.error_counts[model_name]
            error_rate = (error_count / total_requests * 100) if total_requests > 0 else 0
            
            # Log to MLflow if model exists in config
            try:
                if model_name in MODEL_CONFIG:
                    run_id = MODEL_CONFIG[model_name]['run_id']
                    with mlflow.start_run(run_id=run_id):
                        # 1. Log class counts (for Class Distribution)
                        for cls in CLASS_MAPPING:
                            mlflow.log_metric(f"monitoring.class_{cls}_count", 
                                            self.class_counts[model_name][cls])
                        
                        # 2. Log error metrics (for Error Rate)
                        mlflow.log_metric("monitoring.error_count", error_count)
                        mlflow.log_metric("monitoring.error_rate", error_rate)
                        
                        # 3. Log latency metrics (for Average Latency)
                        mlflow.log_metric("monitoring.latency", latency)  # Current prediction
                        mlflow.log_metric("monitoring.avg_latency", avg_latency)  # Running average
                        
                        # 4. Log confidence metrics (for Model Performance)
                        mlflow.log_metric("monitoring.confidence", confidence)  # Current prediction
                        mlflow.log_metric("monitoring.avg_confidence", avg_confidence)  # Running average
                        
                        # 5. Store recent errors (up to 5)
                        if error is not None:
                            # Get current errors
                            recent_errors = list(self.metrics["errors"][model_name])[-5:]
                            
                            # Log each recent error
                            for i, err in enumerate(recent_errors):
                                mlflow.log_metric(f"monitoring.recent_error_{i}", str(err["error"]))
                                mlflow.log_metric(f"monitoring.recent_error_time_{i}", err["timestamp"])
            except Exception as e:
                logger.error(f"Error logging to MLflow: {str(e)}")
                
            # Save monitoring data locally for the dashboard
            self._save_monitoring_data()

    def _compute_image_stats(self, image_tensor):
        """Compute basic image statistics from a tensor"""
				# Handle case when there's no image (like during errors)
        if image_tensor is None:
            return {"mean": [0, 0, 0], "median": [0, 0, 0], "stddev": [0, 0, 0], 
            "size": (0, 0)}

				# Convert tensor to PIL Image for analysis
        if isinstance(image_tensor, torch.Tensor):
            if image_tensor.dim() == 4:# Remove batch dimension if present
                image_tensor = image_tensor.squeeze(0)

						# Convert to numpy and then to PIL
            image_np = image_tensor.permute(1, 2, 0).numpy()
						# Normalize to 0-255 range for PIL
            image_np = ((image_np - image_np.min()) / (image_np.max() - image_np.min()) 
            * 255).astype(np.uint8)
            img = Image.fromarray(image_np)
        else:
            img = image_tensor

				# Calculate image statistics
        stat = ImageStat.Stat(img)
        return {
            "mean": stat.mean,
            "median": stat.median,
            "stddev": stat.stddev,
            "size": img.size,
        }

    def _save_monitoring_data(self):
        """Save monitoring data to disk"""
        monitoring_data = {
            "timestamp": datetime.now().isoformat(),
            "class_counts": self.class_counts,
            "request_counts": self.request_counts,
            "error_counts": self.error_counts,
            "avg_latency": {
                model: sum(lats)/max(len(lats), 1)
                for model, lats in self.metrics["latency"].items()
            },
            "avg_confidence": {
                model: sum(confs)/max(len(confs), 1)
                for model, confs in self.metrics["confidence"].items()
            }
        }

        with open(self.monitoring_data_file, 'w') as f:
            json.dump(monitoring_data, f, indent=2)

    def get_monitoring_data(self):
        """Get current monitoring data for dashboard"""
        with self.lock:
            return {
                "timestamp": datetime.now().isoformat(),
                "class_counts": self.class_counts,
                "request_counts": self.request_counts,
                "error_counts": self.error_counts,
                "avg_latency": {
                    model: sum(lats)/max(len(lats), 1)
                    for model, lats in self.metrics["latency"].items()
                },
                "avg_confidence": {
                    model: sum(confs)/max(len(confs), 1)
                    for model, confs in self.metrics["confidence"].items()
                },
                "recent_errors": {
                    model: list(errs)[-5:] for model, errs in 
                    self.metrics["errors"].items()
                }
            }
            
    def load_metrics_from_mlflow(self):
        """Load historical metrics from MLflow for all models"""
        try:
            # Loop through each model and fetch its metrics
            for model_name, config in MODEL_CONFIG.items():
                run_id = config['run_id']
                
                try:
                    # Get the run data from MLflow
                    run = mlflow.get_run(run_id)
                    metrics = run.data.metrics
                    
                    # 1. Process class counts (for Class Distribution)
                    for cls in CLASS_MAPPING:
                        metric_key = f"monitoring.class_{cls}_count"
                        if metric_key in metrics:
                            self.class_counts[model_name][cls] = int(metrics[metric_key])
                    
                    # Calculate total requests by summing class counts
                    total_requests = sum(self.class_counts[model_name].values())
                    self.request_counts[model_name] = total_requests
                    
                    # 2. Get error counts (for Error Rate)
                    if "monitoring.error_count" in metrics:
                        self.error_counts[model_name] = int(metrics["monitoring.error_count"])
                    
                    # 3. Load average latency (for Average Latency)
                    if "monitoring.avg_latency" in metrics:
                        # Add to the deque so it's included in calculations
                        latency_value = metrics["monitoring.avg_latency"]
                        self.metrics["latency"][model_name].append(latency_value)
                    
                    # 4. Load average confidence (for Model Performance)
                    if "monitoring.avg_confidence" in metrics:
                        # Add to the deque so it's included in calculations
                        confidence_value = metrics["monitoring.avg_confidence"]
                        self.metrics["confidence"][model_name].append(confidence_value)
                    
                    # 5. Load any stored errors
                    for i in range(5):  # Load up to 5 recent errors
                        error_key = f"monitoring.recent_error_{i}"
                        timestamp_key = f"monitoring.recent_error_time_{i}"
                        
                        if error_key in metrics and timestamp_key in metrics:
                            error_msg = metrics[error_key]
                            timestamp = metrics[timestamp_key]
                            
                            self.metrics["errors"][model_name].append({
                                "error": error_msg,
                                "timestamp": timestamp
                            })
                    
                    logger.info(f"Successfully loaded historical metrics for {model_name} from MLflow")
                except Exception as e:
                    logger.error(f"Error loading metrics for {model_name}: {str(e)}")
            
            logger.info("Successfully loaded historical metrics from MLflow")
            return True
        
        except Exception as e:
            logger.error(f"Error loading metrics from MLflow: {str(e)}")
            return False

# Initialize the model monitor
model_monitor = ModelMonitor()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Root endpoint that serves the main UI page"""
    # Pass the list of available models to the template
    return templates.TemplateResponse("ui.html", {"request": request, "models": list(MODEL_CONFIG.keys())})

@app.post("/predict")
async def predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    model_name: str = Form("resnet50"), 
):
    """
    Process the uploaded image and return both prediction and visualization
    
    Parameters:
    - file: The uploaded MRI scan image
    - model_name: The name of the model to use for prediction (defaults to resnet50)
    - background_tasks: FastAPI background tasks object (added for monitoring)
    """
    
    # Start timing the request for monitoring
    start_time = time.time()
    error = None
    
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
        error = str(e)
        # Calculate latency even for failed requests
        latency = (time.time() - start_time) * 1000
        
        # Record error in monitoring (in background)
        background_tasks.add_task(
            model_monitor.record_prediction,
            model_name=model_name, 
            image_tensor=None,
            prediction_idx=-1,  # Invalid prediction
            confidence=0.0,
            latency=latency,
            error=error
        )
        
        return JSONResponse(
            content={"error": f"Error processing request: {error}"}, 
            status_code=400
        )
    
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
    
    # Calculate latency
    latency = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    # Record this prediction in monitoring (in background)
    background_tasks.add_task(
        model_monitor.record_prediction,
        model_name=model_name, 
        image_tensor=input_tensor,
        prediction_idx=prediction_idx,
        confidence=prediction_prob,
        latency=latency
    )
    
    # Return the prediction and visualization data
    return {
        "prediction": prediction_idx,
        "prediction_name": prediction_name,
        "confidence": f"{prediction_prob:.2f}%",
        "model_used": model_name,
        "visualization": visualization_data,
        "latency_ms": f"{latency:.2f}"  # Added latency to response
    }

@app.get("/monitoring", response_class=HTMLResponse)
async def get_monitoring_dashboard(request: Request):
    """Serve the monitoring dashboard"""
    return templates.TemplateResponse("monitoring.html", {"request": request})

@app.get("/api/monitoring")
async def get_monitoring_data():
    """API endpoint to get monitoring data"""
    return model_monitor.get_monitoring_data()

