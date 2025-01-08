import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.model_selection import train_test_split

# Read the data
df = pd.read_csv('data/spam.csv', encoding='latin-1')

# Removing unnecassary columns
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# Renaming columns
df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

# Converting label to numeric values (0 for ham (not spam), 1 for spam)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Determining X and y
X = df['message']
y = df['label']

# Converting text data to numeric format
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Splitting data to train and test pieces
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining model
model = MultinomialNB()

# Training model
model.fit(X_train, y_train)

# Making predictions based on data
model_prediction = model.predict(X_test)

# Main function to simulate
def main():
    """Main function for simulating our model."""

    # Taking user input
    user_input = input("Enter your message: ").split(", ")

    # Vectorizing user input
    transformed_user_input = vectorizer.transform(user_input)

    # Predicticting output based on user input
    output_prediction = model.predict(transformed_user_input)

    # Determining status based on prediction's numerical value
    status = "Spam" if output_prediction == 1 else "Not Spam"

    # Displaying output
    print(f"Message: {user_input} --> Prediction: {status}")
    print()
    
    # Displaying model metrics
    accuracy = accuracy_score(y_test, model_prediction)
    
    print("Model performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Classification Report: {classification_report(y_test, model_prediction)}")
    print(f"Confusion Matrix: {confusion_matrix(y_test, model_prediction)}")


if __name__ == '__main__':
    main()
