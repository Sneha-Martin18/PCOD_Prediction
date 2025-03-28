import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive 'Agg'

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import joblib
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize models
predictor = None

def get_plot():
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

def load_and_train_models():
    global predictor
    try:
        # Load datasets
        slr_data = pd.read_csv('pcod_slr_dataset.csv')
        mlr_data = pd.read_csv('pcod_mlr_dataset.csv')
        poly_data = pd.read_csv('pcod_polynomial_regression_dataset.csv')
        knn_data = pd.read_csv('pcod_knn_dataset.csv')
        log_data = pd.read_csv('pcod_logistic_regression_dataset.csv')
        
        # Initialize models
        slr_model = LinearRegression()
        poly_features = PolynomialFeatures(degree=3)
        poly_model = LinearRegression()
        knn_model = KNeighborsClassifier(n_neighbors=3)
        log_model = LogisticRegression()
        mlr_model = LinearRegression()
        scaler = StandardScaler()
        
        # Train Simple Linear Regression
        X_slr = slr_data[['BMI']]
        y_slr = slr_data['PCOD_Risk_Score']
        slr_model.fit(X_slr, y_slr)
        slr_r2 = r2_score(y_slr, slr_model.predict(X_slr))
        slr_mse = mean_squared_error(y_slr, slr_model.predict(X_slr))
        
        # Train Polynomial Regression
        X_poly = poly_data[['BMI']]
        y_poly = poly_data['PCOD_Risk_Score']
        X_poly_transformed = poly_features.fit_transform(X_poly)
        poly_model.fit(X_poly_transformed, y_poly)
        poly_r2 = r2_score(y_poly, poly_model.predict(X_poly_transformed))
        poly_mse = mean_squared_error(y_poly, poly_model.predict(X_poly_transformed))
        
        # Train KNN and Logistic Regression
        X_knn = knn_data.drop('PCOD_Status', axis=1)
        y_knn = knn_data['PCOD_Status']
        X_knn_scaled = scaler.fit_transform(X_knn)
        
        # Split data for accuracy calculation
        X_train, X_test, y_train, y_test = train_test_split(X_knn_scaled, y_knn, test_size=0.2, random_state=42)
        
        knn_model.fit(X_train, y_train)
        knn_acc = accuracy_score(y_test, knn_model.predict(X_test))
        
        log_model.fit(X_train, y_train)
        log_acc = accuracy_score(y_test, log_model.predict(X_test))
        
        # Train Multiple Linear Regression
        X_mlr = mlr_data.drop('PCOD_Risk_Score', axis=1)
        y_mlr = mlr_data['PCOD_Risk_Score']
        mlr_model.fit(X_mlr, y_mlr)
        mlr_r2 = r2_score(y_mlr, mlr_model.predict(X_mlr))
        mlr_mse = mean_squared_error(y_mlr, mlr_model.predict(X_mlr))
        
        # Store models and metrics
        predictor = {
            'slr': (slr_model, slr_r2, slr_mse),
            'poly': (poly_model, poly_features, poly_r2, poly_mse),
            'knn': (knn_model, scaler, knn_acc),
            'logistic': (log_model, log_acc),
            'mlr': (mlr_model, mlr_r2, mlr_mse)
        }
        
        return True
    except Exception as e:
        print(f"Error training models: {str(e)}")
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if predictor is None:
            if not load_and_train_models():
                return jsonify({
                    'success': False,
                    'error': 'Failed to load models'
                })
        
        data = request.get_json()
        
        # Get input values
        bmi = float(data.get('bmi', 0))
        age = float(data.get('age', 0))
        insulin = float(data.get('insulin', 0))
        glucose = float(data.get('glucose', 0))
        menstrual = int(data.get('menstrual', 0))
        hirsutism = float(data.get('hirsutism', 0))
        testosterone = float(data.get('testosterone', 0))
        
        # Make predictions
        # Simple Linear Regression
        slr_model, slr_r2, slr_mse = predictor['slr']
        slr_pred = slr_model.predict([[bmi]])[0]
        
        # Polynomial Regression
        poly_model, poly_features, poly_r2, poly_mse = predictor['poly']
        X_poly = poly_features.transform([[bmi]])
        poly_pred = poly_model.predict(X_poly)[0]
        
        # KNN and Logistic Regression input
        X_knn = np.array([[age, bmi, insulin, glucose, menstrual, hirsutism, testosterone]])
        X_knn_scaled = predictor['knn'][1].transform(X_knn)
        
        # KNN prediction
        knn_model, _, knn_acc = predictor['knn']
        knn_pred = knn_model.predict(X_knn_scaled)[0]
        
        # Logistic Regression prediction
        log_model, log_acc = predictor['logistic']
        log_pred = log_model.predict(X_knn_scaled)[0]
        
        # Multiple Linear Regression prediction
        mlr_model, mlr_r2, mlr_mse = predictor['mlr']
        mlr_pred = mlr_model.predict(X_knn)[0]
        
        return jsonify({
            'success': True,
            'predictions': {
                'slr_risk_score': float(slr_pred),
                'poly_risk_score': float(poly_pred),
                'knn_prediction': int(knn_pred),
                'logistic_prediction': int(log_pred),
                'mlr_risk_score': float(mlr_pred)
            },
            'accuracy_scores': {
                'slr': {
                    'r2': float(slr_r2),
                    'mse': float(slr_mse)
                },
                'poly': {
                    'r2': float(poly_r2),
                    'mse': float(poly_mse)
                },
                'knn': {
                    'accuracy': float(knn_acc)
                },
                'logistic': {
                    'accuracy': float(log_acc)
                },
                'mlr': {
                    'r2': float(mlr_r2),
                    'mse': float(mlr_mse)
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    load_and_train_models()
    app.run(debug=True) 