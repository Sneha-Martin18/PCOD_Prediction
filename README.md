# PCOD Prediction System

A web application that predicts PCOD (Polycystic Ovarian Disease) using multiple machine learning algorithms.

## Features

- Simple Linear Regression
- Multiple Linear Regression
- Polynomial Regression
- Logistic Regression
- K-Nearest Neighbors (KNN)

## Setup Instructions

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## Dataset Information

The application uses the following datasets:
- pcod_slr_dataset.csv: For Simple Linear Regression
- pcod_mlr_dataset.csv: For Multiple Linear Regression
- pcod_polynomial_regression_dataset.csv: For Polynomial Regression
- pcod_logistic_regression_dataset.csv: For Logistic Regression
- pcod_knn_dataset.csv: For K-Nearest Neighbors

## Usage

1. Fill in the required patient information in the form
2. Select the algorithm you want to use
3. Click "Predict" to get the results
4. View the prediction and accuracy scores

## Model Accuracy

All models are trained to achieve accuracy scores between 0.9 and 1.0 (90-100% accuracy). 