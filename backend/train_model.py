import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Loading dataset...")
# 1. Load the extracted coordinate data
try:
    df = pd.read_csv('medical_signs_data.csv')
except FileNotFoundError:
    print("Error: 'medical_signs_data.csv' not found. Make sure it is in the same folder.")
    exit()

# 2. Separate Features (X) and Target Labels (y)
# 'y' is the word we want to predict (the label column)
# 'X' is the 63 coordinates (x0 to z20)
X = df.drop('label', axis=1)
y = df['label']

print(f"Dataset loaded! Found {len(df)} total frames across {len(y.unique())} different signs.")

# 3. Split the data
# 80% of the data will be used to train the AI. 
# 20% will be hidden and used to test if the AI actually learned it.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train the Model
print("Training the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 5. Evaluate the Accuracy
print("Testing the model's accuracy on unseen data...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n====================================")
print(f" Model Accuracy: {accuracy * 100:.2f}%")
print(f"====================================\n")

print("Detailed Classification Report:")
print(classification_report(y_test, y_pred))

# 6. Save the trained model to a file
# This is the "brain" we will plug into the FastAPI server
model_filename = 'medical_sign_model.pkl'
joblib.dump(model, model_filename)

print(f"\nSuccess! Model saved as '{model_filename}'.")
print("Your AI is now ready to be connected to the app!")