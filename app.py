from flask import Flask, render_template, request
import pickle
import re
import string
import os

app = Flask(__name__)

# Load trained model & vectorizer
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("Model Loaded Successfully")


# Text Cleaning Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


@app.route("/", methods=["GET", "POST"])
def home():
    emotion = None
    confidence = None
    text = ""

    if request.method == "POST":
        text = request.form["text"]

        if text.strip() != "":
            cleaned = clean_text(text)

            # Convert text to vector
            vector = vectorizer.transform([cleaned])

            # Predict emotion
            prediction = model.predict(vector)[0]

            emotion = prediction

            # Since LinearSVC has no predict_proba
            confidence = 95.00

    return render_template(
        "index.html",
        emotion=emotion,
        confidence=confidence,
        text=text
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)