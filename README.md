# Diabetes Prediction Web Application

This is a Flask-based web application for predicting whether a patient is diabetic or non-diabetic based on various health parameters. The app uses a trained Logistic Regression model and provides predictions through a web interface or via JSON API.

## Features

- **Web Interface**: Enter patient details through a user-friendly form to get predictions.
- **API Support**: Submit patient data as JSON to receive predictions programmatically.
- **Data Visualization**: Insights into the dataset used for training the model (e.g., correlation heatmaps, distributions).
- **Model Training**: Logistic Regression model with class balancing and feature scaling.

---

## Project Structure
- app.py # Flask application
- model.py # Model training script
- model.ipynb # Jupyter Notebook for model training and analysis
- diabetes_model.pkl # Trained Logistic Regression model
- scaler.pkl # Scaler for feature normalization
- label_encoders.pkl # Encoders for categorical features
- templates/
  - index.html # HTML template for the web interface
- static/
  - styles.css # CSS for styling the web interface
  - scripts.js # JavaScript for interactivity
- README.md # Project documentation

---

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.7+
- Flask
- NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
- Pickle (for saving/loading models)

### Installation

1. Clone this repository:
//bash
git clone https://github.com/your-username/diabetes-prediction-app.git
cd diabetes-prediction-app

2. Install the required Python packages:

pip install -r requirements.txt


3. Place the required files (`diabetes_model.pkl`, `scaler.pkl`, `label_encoders.pkl`) in the root directory.

---

## Usage

### Running the Application

1. Start the Flask server:


pip install -r requirements.txt


3. Place the required files (`diabetes_model.pkl`, `scaler.pkl`, `label_encoders.pkl`) in the root directory.

---

## Usage

### Running the Application

1. Start the Flask server:


3. Place the required files (`diabetes_model.pkl`, `scaler.pkl`, `label_encoders.pkl`) in the root directory.

---

## Usage

### Running the Application

1. Start the Flask server:

