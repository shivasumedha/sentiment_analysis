import pandas as pd
import pickle
import re
import string
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# -------- Load Dataset --------
df = pd.read_csv("Emotion_classify_data.csv")

# Clean column names
df.columns = df.columns.str.lower().str.strip()

# Force rename if 2 columns
if len(df.columns) == 2:
    df.columns = ["text", "label"]

print("Columns detected:", df.columns)
print("\nClass Distribution:\n", df["label"].value_counts())

# -------- Text Cleaning --------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

df["text"] = df["text"].apply(clean_text)

# -------- Train/Test Split --------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# -------- TF-IDF (Improved) --------
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1,3),
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------- SVM Model (Stronger than Logistic) --------
model = SVC(
    kernel="linear",
    probability=True,
    class_weight="balanced"
)

model.fit(X_train_vec, y_train)

# -------- Accuracy --------
pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, pred)
print("\nAccuracy:", accuracy)

# -------- Save Model --------
os.makedirs("model", exist_ok=True)

pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("Model and Vectorizer saved successfully!")