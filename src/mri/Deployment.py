from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import io
from config import Config

from mri.Model import *


#! REST API: representational state transfer application programming interface
#? It's a method for softwares to communicate over the internet using HTTP methods.

#! Need to display class name alongside class index


#* Load the model:
model = Resnet50(num_classes=4)
model_path = Config.MODEL_PATH
model = torch.load(model_path, map_location=torch.device("cpu"))
model.eval()


app = FastAPI()
templates = Jinja2Templates(directory=str(Config.PROJECT_ROOT / 'Templates'))
    
    
@app.get("/", response_class=HTMLResponse)   #? Handles get requests at the root URL '/', and 
                                             #? sends back a HTML file as a response (browser compatible)
async def read_root(request: Request):    #! this endpoint expects a input of the type: Request object
    
    return templates.TemplateResponse("ui.html", {"request": request})
    #! This renders (process HTML file into a webpage) our 'index.html' file
    
    
    
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Expects an image file to be uploaded.
    The image is processed through a transformation pipeline before being passed to the model.
    """
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        return {"error": f"Invalid image file: {str(e)}"}
    
    processed_image = test_transforms(image)
    
    input_tensor = processed_image.unsqueeze(0)  #* Makes the tensor batch format (1, 224, 224, 3)
    
    with torch.no_grad():
        output = model(input_tensor)
        
    prediction = output.argmax(dim=1).item()
    return {"prediction": prediction}   #* JSON format
        
        
           
    
    
     
