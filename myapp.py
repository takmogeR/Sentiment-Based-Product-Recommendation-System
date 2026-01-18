from flask import Flask, request, render_template
import joblib
import pandas as pd
import re

# =========================================
# Load Model & Vectorizer
# =========================================
model = joblib.load("sentiment_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# =========================================
# Load Dataset
# =========================================
df = pd.read_csv("Dataset-SA.csv")

df = df[['product_name', 'product_price', 'Rate', 'Review', 'Sentiment']].dropna()

df['product_price'] = pd.to_numeric(df['product_price'], errors='coerce')
df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')
df['Sentiment'] = df['Sentiment'].str.lower()

# =========================================
# Product Keyword List (Domain Intelligence)
# =========================================
PRODUCT_KEYWORDS = [
    "air cooler", "cooler",
    "fan",
    "air conditioner", "ac",
    "laptop",
    "mobile", "smartphone",
    "headphone", "earphone",
    "bluetooth", "speaker",
    "washing machine",
    "refrigerator", "fridge",
    "television", "tv"
]

# =========================================
# Detect Product Category from User Input
# =========================================
def extract_product_keyword(text):
    text = text.lower()
    for keyword in PRODUCT_KEYWORDS:
        if re.search(r"\b" + re.escape(keyword) + r"\b", text):
            return keyword
    return None

# =========================================
# Flask App
# =========================================
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    review = request.form["review"]

    # ===============================
    # 1. Predict Sentiment
    # ===============================
    review_vec = vectorizer.transform([review])
    pred_label = model.predict(review_vec)[0]
    pred_prob = model.predict_proba(review_vec).max()

    sentiment = "Positive" if pred_label == 1 else "Negative"
    confidence = round(pred_prob * 100, 2)

    # ===============================
    # 2. Detect Product Type
    # ===============================
    product_keyword = extract_product_keyword(review)

    # ===============================
    # 3. Sentiment Based Filtering
    # ===============================
    if sentiment == "Positive":
        recommendations = df[
            (df["Sentiment"] == "positive") &
            (df["Rate"] >= 4.0)
        ]
    else:
        recommendations = df[
            (df["Sentiment"] == "negative") &
            (df["Rate"] <= 2.5)
        ]

    # ===============================
    # 4. Product Category Filtering
    # ===============================
    if product_keyword:
        recommendations = recommendations[
            recommendations["product_name"].str.lower().str.contains(product_keyword)
        ]

    # ===============================
    # 5. Ranking Logic
    # ===============================
    recommendations = recommendations.sort_values(
        by=["Rate", "product_price"],
        ascending=[False, True]
    )

    recommendations = recommendations.drop_duplicates(
        subset="product_name"
    ).head(5)

    # ===============================
    # 6. Build Output Table
    # ===============================
    if recommendations.empty:
        table_html = "<p style='color:red; font-weight:bold;'>No matching product recommendations found.</p>"
    else:
        table_html = recommendations[
            ['product_name', 'product_price', 'Rate', 'Sentiment']
        ].to_html(
            classes="table table-striped table-bordered",
            header=True,
            index=False
        )

    return render_template(
        "index.html",
        prediction_text=f"Predicted Sentiment: {sentiment}",
        confidence_text=f"Confidence: {confidence}%",
        product_text=f"Detected Product: {product_keyword if product_keyword else 'General'}",
        table_html=table_html
    )

if __name__ == "__main__":
    app.run(debug=True)
