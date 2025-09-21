import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

# Initialize the flask app
app = Flask(__name__)

# Load the model and column transformer
model = pickle.load(open('Model/model.pkl', 'rb'))
ct = pickle.load(open('Model/ct.pkl', 'rb'))

@app.route('/')
def home():
    """Renders the home page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives form data, transforms it, and makes a prediction.
    """
    # Extract form data and convert to appropriate types
    features = [
        int(request.form['Age']),
        int(request.form['City_Tier']),
        request.form['Job_Type'],
        int(request.form['Monthly_Income']),
        int(request.form['EMI_Amount']),
        int(request.form['Active_Loans']),
        int(request.form['Credit_Utilization(%)']),
        int(request.form['UPI_Transactions']),
        int(request.form['Bill_Payment_History(%)']),
        int(request.form['Bank_Balance_Variance']),
        int(request.form['Is_MSME']),
        int(request.form['GST_Revenue']),
        int(request.form['Invoice_Defaults'])
    ]

    # Create a DataFrame from the input features
    # Column names must match those used during training
    column_names = [
        'Age', 'City_Tier', 'Job_Type', 'Monthly_Income', 'EMI_Amount',
        'Active_Loans', 'Credit_Utilization(%)', 'UPI_Transactions',
        'Bill_Payment_History(%)', 'Bank_Balance_Variance', 'Is_MSME',
        'GST_Revenue', 'Invoice_Defaults'
    ]
    
    input_df = pd.DataFrame([features], columns=column_names)
    
    # Transform the input data using the loaded column transformer
    transformed_features = ct.transform(input_df)
    
    # Make a prediction
    prediction = model.predict(transformed_features)
    
    # Get the predicted score and round it
    predicted_score = round(prediction[0])
    
    # Render the result page with the prediction
    return render_template('result.html', prediction_text=f'Predicted Credit Score: {predicted_score}')

if __name__ == "__main__":
    app.run(debug=True)