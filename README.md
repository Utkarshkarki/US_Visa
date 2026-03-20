# US Visa Approval Prediction: End-to-End ML Web Application

## Project Goal
This project is an **End-to-End Machine Learning Web Application** designed to predict the likelihood of a **US Work Visa** application being approved or denied. It leverages machine learning algorithms to analyze an applicant's profile—including education level, job experience, company details, prevailing wages, and region of employment—to generate a real-time prediction.

---

## 🏗️ Project Architecture & Structure

This software is built following industry-standard **MLOps** principles. The architecture is modularly separated into distinct pipelines ensuring maintainability, scalability, and ease of deployment.

### 🌳 Directory Graph
```text
ProjectClassify/
├── artifact/              # Output hub (Model pickles, datasets, logs - gitignored)
├── config/                # Environment configuration YAMLs
├── Notebook/              # Jupyter experiments and drift analysis algorithms
├── static/                # CSS and static browser assets
├── templates/             # HTML front-end pages (usvisa.html)
├── us_visa/               # Main source package
│   ├── components/        # Core ML execution scripts (Ingestion -> Pusher)
│   ├── configuration/     # Cloud and DB connection logic (AWS S3, MongoDB)
│   ├── constants/         # Static global variables
│   ├── data_access/       # Data loading and saving interfaces
│   ├── entity/            # State dataclasses (Artifacts, Config outputs)
│   ├── exception/         # Centralized custom exception handling
│   ├── logger/            # Execution logging configurations
│   ├── pipline/           # Aggregators for Training & Prediction pipelines
│   └── utils/             # Reusable helper functions (file I/O, etc.)
├── app.py                 # FastAPI Backend Web Server
├── demo.py                # Pipeline execution script for retraining
├── requirements.txt       # Python package dependencies
├── setup.py               # Project packaging configuration
└── Dockerfile             # Container instructions for isolated deployment
```

### 1. Data Processing & Training Pipeline (`us_visa/components/`)
The backbone of the project handles the complete lifecycle of standard Machine Learning operations:
* **`data_ingestion.py`**: Handles downloading, extracting, and logically splitting the initial raw data into distinct train and test sets.
* **`data_validation.py`**: Ensures the incoming data maintains data integrity, checking column schemas and formats against expected configurations to prevent upstream errors.
* **`data_transformation.py`**: Prepares raw features for the ML model. It encodes categorical features (like Education or Continent), scales numerical data, and handles missing values.
* **`model_trainer.py`**: Trains various classification algorithms, dynamically evaluates them against scoring metrics, and saves the highest-performing model.
* **`model_evaluation.py` & `model_pusher.py`**: Evaluates new models against previously successful ones and securely pushes the newly trained model (as a pickle file) into **AWS S3 Cloud Storage**.

### 2. Prediction Pipeline & Web App
* **`us_visa/pipline/prediction_pipeline.py`**: Contains the `USvisaData` and `USvisaClassifier` classes. This pipeline safely retrieves the active model weights from AWS S3, constructs the input dataframe, and processes single HTTP requests.
* **FastAPI Backend (`app.py`)**: The `app.py` script runs a FastAPI web server. It handles `GET` requests by serving our frontend UI, and `POST` requests to grab applicant parameters from the HTML form, process it through our prediction pipeline, and return the visa decision.
* **Jinja Frontend (`templates/usvisa.html` & `static/`)**: A responsive, bootstrap-styled HTML interface allowing users to input applicant features naturally.

---

## 🛠️ Tech Stack & MLOps Edges
* **Algorithm/Machine Learning**: Supervised Classification Algorithms (handled via `scikit-learn` / Python)
* **Web Framework**: **FastAPI** (with `uvicorn` and Jinja2Templates)
* **MLOps / Data Drift Monitoring**: **Evidently AI** is configured to monitor incoming data distributions and track performance decay/data drifting over time.
* **Storage / Cloud**: Active predictive models are managed via **Amazon Web Services (AWS S3)**.
* **Environment Control**: Setuptools & requirements files configured for Python isolated Virtual Environments padding dependency conflicts.

---

## 🚀 How to Run the Project Locally

### 1. Setup Environment
Ensure you have cloned the repository, and are in the root Git directory (`ProjectClassify/ProjectClassify`). Open a terminal (like Git Bash) and run:
```bash
# Create a virtual environment named 'visa'
python -m venv visa

# Activate it (on Windows Git Bash)
source visa/Scripts/activate

# Install the project and dependencies
pip install -r requirements.txt
```

### 2. Configure AWS (For Model Storage)
Since the prediction system dynamically fetches models from cloud storage, ensure your AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) are appropriately set in your local system's environment variables.

### 3. Run the Training Pipeline (Optional)
If you need to retrain the ML model from scratch on new data, execute:
```bash
python demo.py
```
*Wait for data ingestion, training, and S3 upload to complete.*

### 4. Run the Web Application
To serve the frontend user interface and process predictions:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8080
```
Open your internet browser and navigate to: **http://localhost:8080**
