from joblib import load
import numpy as np

# Load the saved model created in model.py
model = load("C:\Armanezamento\Challenges\ProzisClassificationAPI\model\classifier.joblib")

# Function to predict the intent and calculate the confidence score
def predict_intent(text: str):
    pred = model.predict([text])[0]
    prob = np.max(model.predict_proba([text]))
    return pred, float(prob)