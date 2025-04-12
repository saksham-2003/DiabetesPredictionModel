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

python app.py

text

2. Open your browser and navigate to:

http://127.0.0.1:5000/

text

3. Enter patient details in the form and click **Predict** to see results.

### API Usage

You can also use the `/predict` endpoint to get predictions via JSON:

- **Endpoint**: `POST /predict`
- **Request Body (JSON)**:

{
"gender": 1,
"age": 45,
"hypertension": 0,
"heart_disease": 0,
"smoking_history": 2,
"bmi": 25.5,
"HbA1c_level": 6.5,
"blood_glucose_level": 140
}

text
- **Response**:

{
"prediction": 1,
"result": "Diabetic"
}

text

---

## Model Training

The model is trained using the `model.py` script or `model.ipynb` notebook:

1. Load the dataset (`diabetes_prediction_dataset.csv`).
2. Encode categorical features (`gender`, `smoking_history`).
3. Scale numerical features using `MinMaxScaler`.
4. Train a Logistic Regression model with class balancing.
5. Evaluate performance (accuracy, classification report).
6. Save the trained model, scaler, and encoders using `pickle`.

---

## Screenshots

### Web Interface

![Web Interface](https://via.placeholder.com/800x400.png?text=Web+Interface+Screenshot)

### Data Visualizations

![Data Visualization](https://via.placeholder.com/800x400.png?text=Data+Visualization+Screenshot)

---

## Technologies Used

- **Backend**: Flask, Python
- **Frontend**: HTML, CSS (with `styles.css`), JavaScript (`scripts.js`)
- **Machine Learning**: Scikit-learn (Logistic Regression)
- **Data Visualization**: Matplotlib, Seaborn

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork this repository.
2. Create a new branch:

git checkout -b feature-name

text
3. Commit your changes:

git commit -m "Add new feature"

text
4. Push to your branch:

git push origin feature-name

text
5. Open a pull request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Special thanks to:

- [Scikit-learn](https://scikit-learn.org/) for machine learning tools.
- [Flask](https://flask.palletsprojects.com/) for web development.
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for data visualization.

---
