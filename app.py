from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import os
import sys

# To ensure the us_visa module can be imported properly if run locally
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from us_visa.pipline.prediction_pipeline import USvisaData, USvisaClassifier

app = FastAPI()

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Renders the frontend HTML form from templates/usvisa.html
    """
    return templates.TemplateResponse("usvisa.html", {"request": request, "context": "Waiting for Input..."})

@app.post("/", response_class=HTMLResponse)
async def predict_route(
    request: Request,
    continent: str = Form(...),
    education_of_employee: str = Form(...),
    has_job_experience: str = Form(...),
    requires_job_training: str = Form(...),
    no_of_employees: int = Form(...),
    region_of_employment: str = Form(...),
    prevailing_wage: float = Form(...),
    unit_of_wage: str = Form(...),
    full_time_position: str = Form(...),
    company_age: int = Form(...)
):
    """
    Receives form data, uses the prediction pipeline to get result, and updates the HTML.
    """
    try:
        usvisa_data = USvisaData(
            continent=continent,
            education_of_employee=education_of_employee,
            has_job_experience=has_job_experience,
            requires_job_training=requires_job_training,
            no_of_employees=no_of_employees,
            region_of_employment=region_of_employment,
            prevailing_wage=prevailing_wage,
            unit_of_wage=unit_of_wage,
            full_time_position=full_time_position,
            company_age=company_age
        )

        usvisa_df = usvisa_data.get_usvisa_input_data_frame()
        
        usvisa_classifier = USvisaClassifier()
        prediction = usvisa_classifier.predict(usvisa_df)
        
        # Handle array-like or single item return values safely
        status = prediction[0] if isinstance(prediction, (list, tuple)) or hasattr(prediction, "tolist") else prediction

        # Formulate a clean context message
        context = f"Approved" if str(status) == "1" else f"Denied" if str(status) == "0" else str(status)

        return templates.TemplateResponse("usvisa.html", {"request": request, "context": context})
    
    except Exception as e:
        return templates.TemplateResponse("usvisa.html", {"request": request, "context": f"Error: {str(e)}"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
