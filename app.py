from flask import Flask, render_template, request
import pickle
import re
import string

app = Flask(__name__)

# Load trained model & vectorizer
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

print("Model Loaded Successfully")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

@app.route("/", methods=["GET", "POST"])
def home():
    emotion = None
    confidence = None
    text = ""

    if request.method == "POST":
        text = request.form["text"]

        if text.strip() != "":
            cleaned = clean_text(text)
            vector = vectorizer.transform([cleaned])

            prediction = model.predict(vector)[0]
            probabilities = model.predict_proba(vector)

            emotion = prediction
            confidence = round(max(probabilities[0]) * 100, 2)

    return render_template(
        "index.html",
        emotion=emotion,
        confidence=confidence,
        text=text
    )

if __name__ == "__main__":
    app.run(debug=True)