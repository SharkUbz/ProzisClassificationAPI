from pydantic import BaseModel

# This defines a model which requires a string with the user's text
class TextInput(BaseModel):
    text: str

# This defines a model which requires a string and a float which will show the intent and confidence score the system
# has regarding the user's text
class IntentOutput(BaseModel):
    intent: str
    confidence_score: float