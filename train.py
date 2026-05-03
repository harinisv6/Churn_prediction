import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression    
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler

# 1. LOAD DATA

df = pd.read_excel("data/Telco_customer_churn.xlsx")

# 2. CLEAN DATA

# Remove duplicates

df.drop_duplicates(inplace=True)

# Drop unnecessary columns

df.drop(columns=[
    'CustomerID', 'Count', 'Country', 'State', 'City',
    'Zip Code', 'Lat Long', 'Latitude', 'Longitude',
    'Senior Citizen', 'Partner', 'Payment Method',
    'Churn Label', 'Churn Score', 'CLTV',
    'Churn Reason', 'Device Protection',
    'Online Backup', 'Online Security',
    'Multiple Lines', 'Phone Service',
    'Dependents'
], inplace=True)

# Fix datatype

df['Total Charges'] = pd.to_numeric(
    df['Total Charges'],
    errors='coerce'
)

# Handle missing values

df['Total Charges'] = df['Total Charges'].fillna(
    df['Total Charges'].mean()
)

# 3. SPLIT FEATURES AND TARGET

y = df['Churn Value']
X = df.drop(columns=['Churn Value'])

# 4. ENCODE CATEGORICAL DATA

X = pd.get_dummies(X, drop_first=True)

# Save feature names for dashboard

feature_columns = X.columns

# 5. TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 6. SCALE DATA

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. TRAIN MODEL

model = LogisticRegression(
    max_iter=2000,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# 8. PREDICTION

y_pred = model.predict(X_test)

# 9. EVALUATION

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 10. SAVE FILES

joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")

print("\nFiles saved ✅")
print("churn_model.pkl")
print("scaler.pkl")
print("feature_columns.pkl")
