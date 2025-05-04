import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from joblib import dump

#Read the data
with open("C:\Armanezamento\Challenges\ProzisClassificationAPI\data\intent_dataset.json", "r", encoding="utf-8") as f:
    data = pd.read_json(f)

# X for input features and y for target labels
X = data["text"]
y = data["label"]

# Split the data into train and test subsets (80% for training and 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# The MLP classifier can be changed with others referenced in the Imports section
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MLPClassifier(random_state=1, max_iter=300))
])

#Training the model
classifier = pipeline.fit(X_train, y_train)
prediction = classifier.predict(X_test)

#Model's accuracy, precision, recall, f1-score, support and confusion matrix
print("\nAccuracy:", accuracy_score(y_test,prediction))
print("\n", classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))

# Save the model (We can use the package pickle as well)
#dump(pipeline, "C:\Armanezamento\Challenges\ProzisClassificationAPI\model\classifier.joblib")