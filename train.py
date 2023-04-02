import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Load the telco customer churn dataset
df = pd.read_csv('Data.csv')

# Convert categorical variables to dummy variables
df = pd.get_dummies(df, columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                 'PaymentMethod'])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('Churn', axis=1), df['Churn'], test_size=0.2)
X_test.to_csv("X_test.csv",index=False)
y_test.to_csv("y_test.csv",index=False)

Scaler = StandardScaler()
X_train = Scaler.fit_transform(X_train)
X_test = Scaler.transform(X_test)
# Train a Logistic Regression classifier on the training set

clf = LogisticRegression()
clf.fit(X_train, y_train)

# Save the trained model
joblib.dump(clf, 'model.pkl')
joblib.dump(Scaler, 'scaler.pkl')

# Evaluate the classifier on the test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Test accuracy: {accuracy}')