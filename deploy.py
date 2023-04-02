# Import necessary libraries
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Create Flask app
app = Flask(__name__)

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

Model = model
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')
X_test = scaler.transform(X_test)

# Define function to train and evaluate the model
def test(X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return model, accuracy, cm, report, y_pred

# Define function to display results
def display_results(model, accuracy, cm, report):
    results = {}
    results['Accuracy'] = accuracy
    results['Confusion Matrix'] = cm.tolist()
    results['Classification Report'] = classification_report
    return results
@app.route('/')
def home():
    return render_template('index.html')
    
# Define main route
@app.route('/predict', methods=['POST'])
def index():

    gender = request.form['gender']
    SeniorCitizen = request.form['SeniorCitizen']
    Partner = request.form['Partner']
    Dependents = request.form['Dependents']
    tenure = request.form['tenure']
    PhoneService = request.form['PhoneService']
    MultipleLines = request.form['MultipleLines']
    InternetService = request.form['InternetService']
    OnlineSecurity = request.form['OnlineSecurity']
    OnlineBackup = request.form['OnlineBackup']
    DeviceProtection = request.form['DeviceProtection']
    TechSupport = request.form['TechSupport']
    StreamingTV = request.form['StreamingTV']
    StreamingMovies = request.form['StreamingMovies']
    Contract = request.form['Contract']
    PaperlessBilling = request.form['PaperlessBilling']
    PaymentMethod = request.form['PaymentMethod']
    MonthlyCharges = request.form['MonthlyCharges']

    data = [[gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges]]
    
    x_test = pd.DataFrame(data, columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges'])

    
    
    x_test = pd.get_dummies(x_test)

    # Predict on the new data
    Y_pred = Model.predict(X_test)[0]

    if request.method == 'POST':
        model, accuracy, cm, report, y_pred = test(X_test, y_test)
        results = display_results(model, accuracy, cm, report)
        if Y_pred == [1]:
            output_text = "This customer is likely to be churned!!"
        else:
            output_text = "This customer is likely to continue!!"

        return render_template('results.html', results=results, output=output_text)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
