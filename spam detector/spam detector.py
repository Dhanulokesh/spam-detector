# 📌 Import necessary libraries
import pandas as pd
import numpy as np
import string
import nltk
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 📌 Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# 📌 Load dataset (Ensure spam.csv is in the correct location)
try:
    df = pd.read_csv("e:/projects/spam detector/spam.csv", encoding='latin-1')
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ ERROR: 'spam.csv' not found. Please place it in 'e:/projects/'")
    exit()

# 📌 Keep only relevant columns and rename them
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Encode spam as 1, ham as 0

# ✅ Check if dataset is loaded correctly
print(df.head())

# 📌 Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# ✅ Define text cleaning function
def clean_text(text):
    if pd.isna(text):  # Handle missing values
        return ""
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()
    
    try:
        words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    except LookupError:  # If stopwords fail to load
        nltk.download('stopwords')
        words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]

    return " ".join(words)

# ✅ Apply text cleaning
df['message'] = df['message'].fillna('')  # Replace NaN with empty strings
df['cleaned_message'] = df['message'].apply(clean_text)

# ✅ Check cleaned messages
print("\n✅ Sample cleaned messages:")
print(df[['message', 'cleaned_message']].head())

# 📌 Feature Extraction using TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_message']).toarray()
y = df['label']

# 📌 Split Dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Train Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train, y_train)
print("✅ Model training completed!")

# 📌 Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.4f}")
print("\n📌 Classification Report:\n", classification_report(y_test, y_pred))

# 📌 Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 📌 Function to Predict New Messages
def predict_spam(message):
    cleaned_msg = clean_text(message)
    vectorized_msg = vectorizer.transform([cleaned_msg]).toarray()
    prediction = model.predict(vectorized_msg)
    return "Spam" if prediction[0] == 1 else "Ham"

# 📌 Example Predictions
print("\n🔍 Example Predictions:")
test_messages = ["Congratulations! You've won a free iPhone!", "Hey, let's meet for lunch."]
for msg in test_messages:
    print(f"📩 Message: {msg} → Prediction: {predict_spam(msg)}")
