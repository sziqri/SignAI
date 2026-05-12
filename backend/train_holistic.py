import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Loading holistic spatial dataset...")
try:
    df = pd.read_csv('holistic_medical_data.csv')
except FileNotFoundError:
    print("Error: 'holistic_medical_data.csv' not found.")
    exit()

# Features (X) will now be 258 columns wide (f0 to f257)
X = df.drop('label', axis=1)
y = df['label']

print(f"Dataset loaded! Training on {len(df)} frames across {len(y.unique())} signs.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the Holistic Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("Testing accuracy...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n====================================")
print(f" Holistic Model Accuracy: {accuracy * 100:.2f}%")
print(f"====================================\n")
print(classification_report(y_test, y_pred))

# Save the new upgraded model
model_filename = 'holistic_medical_model.pkl'
joblib.dump(model, model_filename)

print(f"\nSuccess! Upgraded model saved as '{model_filename}'.")