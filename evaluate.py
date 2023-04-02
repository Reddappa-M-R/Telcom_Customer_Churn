import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from sklearn.preprocessing import StandardScaler
# Load the test set
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')


# Load the trained model
clf = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Preprocess the test set
#X_test = pd.get_dummies(X_test, columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
#                                           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
#                                           'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
#                                           'PaymentMethod'])

# Make predictions on the test set
X_test = scaler.transform(X_test)
y_pred = clf.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f'Test accuracy: {accuracy:.2f}')
print(f'Test precision: {precision:.2f}')
print(f'Test recall: {recall:.2f}')
print(f'Test F1 score: {f1:.2f}')
print(f'Confusion matrix:\n{cm}')
