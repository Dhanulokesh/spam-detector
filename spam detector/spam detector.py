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

nltk.download('stopwords')
nltk.download('wordnet')

try:
    df = pd.read_csv("e:/projects/spam detector/spam.csv", encoding='latin-1')
    print(" Dataset loaded successfully!")
except FileNotFoundError:
    print(" ERROR: 'spam.csv' not found. Please place it in 'e:/projects/'")
    exit()

df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  

print(df.head())

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isna(text):  
        return ""
    text = str(text).lower()  
    text = re.sub(r'\d+', '', text)  
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    words = text.split()
    
    try:
        words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    except LookupError:  
        nltk.download('stopwords')
        words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]

    return " ".join(words)

df['message'] = df['message'].fillna('')  
df['cleaned_message'] = df['message'].apply(clean_text)

print("\n Sample cleaned messages:")
print(df[['message', 'cleaned_message']].head())

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_message']).toarray()
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)
print(" Model training completed!")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Model Accuracy: {accuracy:.4f}")
print("\n Classification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

def predict_spam(message):
    cleaned_msg = clean_text(message)
    vectorized_msg = vectorizer.transform([cleaned_msg]).toarray()
    prediction = model.predict(vectorized_msg)
    return "Spam" if prediction[0] == 1 else "Ham"

print("\n🔍 Example Predictions:")
test_messages = ["Congratulations! You've won a free iPhone!", "Hey, let's meet for lunch."]
for msg in test_messages:
    print(f" Message: {msg} → Prediction: {predict_spam(msg)}")
