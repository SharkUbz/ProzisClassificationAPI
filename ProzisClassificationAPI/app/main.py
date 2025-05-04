from fastapi import FastAPI
from app.schema import TextInput, IntentOutput
from app.model import predict_intent
from fastapi import HTTPException


app = FastAPI()

# Declares a POST endpoint at /classify. Input matching the created TextInput schema and returns a response
# formatted according to the IntentOutput schema
@app.post("/classify", response_model=IntentOutput)
def classify(text_input: TextInput):
    # Checks if the input text is empty or contains only whitespaces.
    if not text_input.text.strip():
        # If the error occurs there will be a message explaining the user
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    intent, score = predict_intent(text_input.text)
    return {"intent": intent, "confidence_score": score}