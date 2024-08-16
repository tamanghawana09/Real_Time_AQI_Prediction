from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
from prophet import Prophet
import matplotlib.pyplot as plt
import io
import base64
import os
from datetime import datetime

app = Flask(__name__)

# Define the paths to the models
prophet_model_filename = os.path.join('ML_Model', 'prophet_model.pkl')
regression_model_filename = os.path.join('ML_Model', 'regression_model.pkl')
classification_model_filename = os.path.join('ML_Model', 'classification_model.pkl')

# Load the pre-trained models
try:
    prophet_model = joblib.load(prophet_model_filename)
    regression_model = joblib.load(regression_model_filename)
    classification_model = joblib.load(classification_model_filename)
except FileNotFoundError as e:
    raise RuntimeError(f"Model file not found: {e}")

def categorize_aqi(aqi):
    """Categorize AQI values into quality categories."""
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Moderate'
    elif aqi <= 150:
        return 'Unhealthy for Sensitive Groups'
    elif aqi <= 200:
        return 'Unhealthy'
    elif aqi <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'

def get_recommendation(category):
    """Provide recommendations based on AQI category."""
    recommendations = {
        'Good': 'Air quality is satisfactory. No action needed.',
        'Moderate': 'Air quality is acceptable; however, there may be a risk for some people. Consider reducing outdoor activities.',
        'Unhealthy for Sensitive Groups': 'Members of sensitive groups may experience health effects. Limit prolonged outdoor exertion.',
        'Unhealthy': 'Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects. Avoid prolonged outdoor exertion.',
        'Very Unhealthy': 'Health alert: everyone may experience more serious health effects. Limit time outside and take precautions.',
        'Hazardous': 'Health warnings of emergency conditions. The entire population is likely to be affected. Stay indoors and avoid any outdoor activities.'
    }
    return recommendations.get(category, 'No recommendation available.')

@app.route('/')
def index():
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return render_template('index.html', current_time=current_time)

@app.route('/forecast', methods=['POST'])
def forecast():
    current_time = datetime.now()

    # Create a DataFrame for the next 24 hours
    future_dates = pd.date_range(start=current_time, periods=24, freq='h')
    future_df = pd.DataFrame(future_dates, columns=['ds'])
    
    # Predict future AQI using Prophet model
    forecast = prophet_model.predict(future_df)
    
    # Categorize AQI values
    forecast['category'] = forecast['yhat'].apply(categorize_aqi)
    forecast['recommendation'] = forecast['category'].apply(get_recommendation)
    
    # Prepare the forecasted AQI data
    forecasted_aqi = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'category', 'recommendation']].to_dict(orient='records')
    
    # Plot the forecast
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecasted AQI', color='green', linestyle='--')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='green', alpha=0.2)
    ax.set_title('24-hour AQI Forecast', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('AQI', fontsize=14)
    ax.legend()
    ax.grid(True)
    
    # Save the plot to a BytesIO object
    forecast_img = io.BytesIO()
    plt.savefig(forecast_img, format='png', bbox_inches='tight')
    forecast_img.seek(0)
    plt.close()

    # Create visualization of AQI categories
    fig, ax = plt.subplots(figsize=(12, 6))
    categories = forecast['category'].value_counts()
    ax.bar(categories.index, categories.values, color='skyblue', edgecolor='black')
    ax.set_title('AQI Categories Distribution', fontsize=16, fontweight='bold')
    ax.set_xlabel('Category', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.grid(axis='y', linestyle='--')
    
    # Save the category plot to a BytesIO object
    category_img = io.BytesIO()
    plt.savefig(category_img, format='png', bbox_inches='tight')
    category_img.seek(0)
    plt.close()
    
    # Encode images to Base64
    forecast_img_base64 = base64.b64encode(forecast_img.getvalue()).decode('utf-8')
    category_img_base64 = base64.b64encode(category_img.getvalue()).decode('utf-8')
    
    return render_template('forecast.html', current_time=current_time, forecasted_aqi=forecasted_aqi, 
                           plot_url='data:image/png;base64,' + forecast_img_base64, 
                           category_plot_url='data:image/png;base64,' + category_img_base64)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Extract input values from the form
        precipitation = float(request.form.get('precipitation', 0))
        temperature = float(request.form.get('temperature', 0))
        relative_humidity = float(request.form.get('relative_humidity', 0))
        pm25 = float(request.form.get('pm25', 0))
        o3 = float(request.form.get('o3', 0))
        
        now = datetime.now()
        day = now.day
        hour = now.hour
        month = now.month
        year = now.year
        
        # Prepare input for regression model
        regression_input = pd.DataFrame([[precipitation, temperature, relative_humidity, pm25, o3, day, hour, month, year]],
                                        columns=['Precipitation', 'Temperature', 'Relative Humidity', 'PM 2.5', 'O3', 'day', 'hour', 'month', 'year'])
        regression_prediction = regression_model.predict(regression_input)[0]
        
        # Prepare input for classification model
        classification_input = pd.DataFrame([[precipitation, temperature, relative_humidity, pm25, o3, day, hour, month, year]],
                                            columns=['Precipitation', 'Temperature', 'Relative Humidity', 'PM 2.5', 'O3', 'day', 'hour', 'month', 'year'])
        classification_prediction = classification_model.predict(classification_input)[0]
        category_name = categorize_aqi(regression_prediction)
        recommendation = get_recommendation(category_name)
        
        return render_template('predict.html', current_time=datetime.now(), 
                               regression_prediction=regression_prediction, 
                               classification_prediction=classification_prediction,
                               category_name=category_name,
                               recommendation=recommendation)
    
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
